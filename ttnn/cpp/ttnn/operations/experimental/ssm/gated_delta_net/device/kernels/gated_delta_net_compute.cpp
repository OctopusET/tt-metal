// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// GatedDeltaNet recurrence: fp32 state via dual-pack trick.
//
// State CBs (cb_state, cb_state_new): fp32 for DRAM precision.
// matmul_tiles works with fp32 CBs (special unpack path).
// Element-wise ops use bf16 intermediates.
//
// Phase 1 "dual pack": decay multiply produces fp32 DST result,
// pack TWICE to fp32 cb_state_new (for matmul) AND bf16 cb_sd2 (for add).
// This gives both fp32 matmul precision and bf16 availability for eltwise.

#include <cstdint>

#include "api/compute/eltwise_binary.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/bcast.h"
#include "api/compute/matmul.h"
#include "api/compute/transpose_wh.h"
#include "api/compute/pack.h"

constexpr uint32_t cb_q = get_compile_time_arg_val(0);
constexpr uint32_t cb_k = get_compile_time_arg_val(1);
constexpr uint32_t cb_v = get_compile_time_arg_val(2);
constexpr uint32_t cb_decay = get_compile_time_arg_val(3);
constexpr uint32_t cb_beta = get_compile_time_arg_val(4);
constexpr uint32_t cb_state = get_compile_time_arg_val(5);      // fp32 input
constexpr uint32_t cb_state_new = get_compile_time_arg_val(6);  // fp32 output
constexpr uint32_t cb_out = get_compile_time_arg_val(7);
constexpr uint32_t cb_sd = get_compile_time_arg_val(8);     // bf16 temp for conversion
constexpr uint32_t cb_tmp = get_compile_time_arg_val(9);    // scratch
constexpr uint32_t cb_tmp2 = get_compile_time_arg_val(10);  // scratch
constexpr uint32_t cb_sd2 = get_compile_time_arg_val(11);   // bf16 decayed state for add
constexpr uint32_t D_TILES = get_compile_time_arg_val(12);
constexpr uint32_t STATE_TILES = get_compile_time_arg_val(13);

void kernel_main() {
    const uint32_t num_heads_this_core = get_arg_val<uint32_t>(0);

    for (uint32_t hh = 0; hh < num_heads_this_core; hh++) {
        cb_wait_front(cb_q, D_TILES);
        cb_wait_front(cb_k, D_TILES);
        cb_wait_front(cb_v, D_TILES);
        cb_wait_front(cb_decay, 1);
        cb_wait_front(cb_beta, 1);
        cb_wait_front(cb_state, STATE_TILES);

        // ============================================================
        // Phase 1a: Convert fp32 state -> bf16 for eltwise decay
        // ============================================================
        cb_reserve_back(cb_sd, STATE_TILES);
        init_sfpu(cb_state, cb_sd);
        for (uint32_t t = 0; t < STATE_TILES; t++) {
            tile_regs_acquire();
            copy_tile(cb_state, t, 0);  // fp32 CB -> fp32 DST
            tile_regs_commit();
            tile_regs_wait();
            pack_tile(0, cb_sd);  // fp32 DST -> bf16 CB
            tile_regs_release();
        }
        cb_push_back(cb_sd, STATE_TILES);
        cb_pop_front(cb_state, STATE_TILES);

        // ============================================================
        // Phase 1b: Decay multiply, DUAL PACK to fp32 + bf16
        // ============================================================
        cb_wait_front(cb_sd, STATE_TILES);
        cb_reserve_back(cb_state_new, STATE_TILES);  // fp32 for matmul
        cb_reserve_back(cb_sd2, STATE_TILES);        // bf16 for add

        binary_op_init_common(cb_sd, cb_decay, cb_sd2);
        mul_tiles_bcast_scalar_init_short(cb_sd, cb_decay);
        for (uint32_t t = 0; t < STATE_TILES; t++) {
            tile_regs_acquire();
            mul_tiles_bcast_scalar(cb_sd, cb_decay, t, 0, 0);  // bf16 * bf16 -> fp32 DST
            tile_regs_commit();
            tile_regs_wait();
            pack_tile(0, cb_state_new);  // fp32 DST -> fp32 CB (full precision)
            pack_reconfig_data_format(cb_sd2);
            pack_tile(0, cb_sd2);  // fp32 DST -> bf16 CB (for eltwise add later)
            pack_reconfig_data_format(cb_state_new);
            tile_regs_release();
        }
        cb_push_back(cb_state_new, STATE_TILES);
        cb_push_back(cb_sd2, STATE_TILES);
        cb_pop_front(cb_sd, STATE_TILES);
        cb_pop_front(cb_decay, 1);

        // ============================================================
        // Phase 2: kv_mem = k @ state_decayed[fp32], delta = (v-kv_mem)*beta
        // matmul_tiles works with fp32 cb_state_new
        // ============================================================
        cb_wait_front(cb_state_new, STATE_TILES);

        cb_reserve_back(cb_tmp, D_TILES);
        mm_init(cb_k, cb_state_new, cb_tmp);
        for (uint32_t j = 0; j < D_TILES; j++) {
            tile_regs_acquire();
            for (uint32_t i = 0; i < D_TILES; i++) {
                matmul_tiles(cb_k, cb_state_new, i, i * D_TILES + j, 0);
            }
            tile_regs_commit();
            tile_regs_wait();
            pack_tile(0, cb_tmp);
            tile_regs_release();
        }
        cb_push_back(cb_tmp, D_TILES);

        // delta = (v - kv_mem) * beta
        cb_wait_front(cb_tmp, D_TILES);
        cb_reserve_back(cb_tmp2, D_TILES);
        binary_op_init_common(cb_v, cb_tmp, cb_tmp2);
        sub_tiles_init(cb_v, cb_tmp);
        for (uint32_t t = 0; t < D_TILES; t++) {
            tile_regs_acquire();
            sub_tiles(cb_v, cb_tmp, t, t, 0);
            tile_regs_commit();
            tile_regs_wait();
            pack_tile(0, cb_tmp2);
            tile_regs_release();
        }
        cb_push_back(cb_tmp2, D_TILES);
        cb_pop_front(cb_v, D_TILES);
        cb_pop_front(cb_tmp, D_TILES);

        cb_wait_front(cb_tmp2, D_TILES);
        cb_reserve_back(cb_tmp, D_TILES);
        binary_op_init_common(cb_tmp2, cb_beta, cb_tmp);
        mul_tiles_bcast_scalar_init_short(cb_tmp2, cb_beta);
        for (uint32_t t = 0; t < D_TILES; t++) {
            tile_regs_acquire();
            mul_tiles_bcast_scalar(cb_tmp2, cb_beta, t, 0, 0);
            tile_regs_commit();
            tile_regs_wait();
            pack_tile(0, cb_tmp);
            tile_regs_release();
        }
        cb_push_back(cb_tmp, D_TILES);
        cb_pop_front(cb_tmp2, D_TILES);
        cb_pop_front(cb_beta, 1);

        // ============================================================
        // Phase 3: state_new = state_decayed + k^T @ delta
        // Pop fp32 state_new (done with matmul), rewrite with updated state.
        // binary_dest_reuse adds bf16 cb_sd2 to fp32 DST from matmul.
        // ============================================================
        cb_wait_front(cb_tmp, D_TILES);
        cb_wait_front(cb_sd2, STATE_TILES);

        // Transpose k
        cb_reserve_back(cb_tmp2, D_TILES);
        transpose_wh_init(cb_k, cb_tmp2);
        for (uint32_t t = 0; t < D_TILES; t++) {
            tile_regs_acquire();
            transpose_wh_tile(cb_k, t, 0);
            tile_regs_commit();
            tile_regs_wait();
            pack_tile(0, cb_tmp2);
            tile_regs_release();
        }
        cb_push_back(cb_tmp2, D_TILES);
        cb_wait_front(cb_tmp2, D_TILES);

        // Pop old state_new, reserve for new
        cb_pop_front(cb_state_new, STATE_TILES);
        cb_reserve_back(cb_state_new, STATE_TILES);

        for (uint32_t i = 0; i < D_TILES; i++) {
            for (uint32_t j = 0; j < D_TILES; j++) {
                uint32_t sd_idx = i * D_TILES + j;

                // Outer product: k_t[i] @ delta[j] -> fp32 DST[0]
                mm_init(cb_tmp2, cb_tmp, cb_state_new);
                tile_regs_acquire();
                matmul_tiles(cb_tmp2, cb_tmp, i, j, 0);
                tile_regs_commit();
                tile_regs_wait();

                // Add decayed state (bf16): DST[0] += cb_sd2[sd_idx]
                // binary_dest_reuse uses llk_unpack_A (SrcA only, works with bf16)
                binary_dest_reuse_tiles_init(cb_sd2);
                binary_dest_reuse_tiles(cb_sd2, sd_idx, 0);

                // Pack to fp32 state_new (preserves fp32 precision from DST)
                pack_tile(0, cb_state_new);
                tile_regs_release();
            }
        }
        cb_push_back(cb_state_new, STATE_TILES);
        cb_pop_front(cb_sd2, STATE_TILES);
        cb_pop_front(cb_tmp, D_TILES);
        cb_pop_front(cb_tmp2, D_TILES);

        // ============================================================
        // Phase 5: output = q @ state_new[fp32]
        // ============================================================
        cb_wait_front(cb_state_new, STATE_TILES);
        cb_reserve_back(cb_out, D_TILES);
        mm_init(cb_q, cb_state_new, cb_out);
        for (uint32_t j = 0; j < D_TILES; j++) {
            tile_regs_acquire();
            for (uint32_t i = 0; i < D_TILES; i++) {
                matmul_tiles(cb_q, cb_state_new, i, i * D_TILES + j, 0);
            }
            tile_regs_commit();
            tile_regs_wait();
            pack_tile(0, cb_out);
            tile_regs_release();
        }
        cb_push_back(cb_out, D_TILES);

        cb_pop_front(cb_q, D_TILES);
        cb_pop_front(cb_k, D_TILES);
    }
}

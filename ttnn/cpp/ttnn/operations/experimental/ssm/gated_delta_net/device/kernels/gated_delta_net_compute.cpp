// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// GatedDeltaNet recurrence: fp32 state, k as column vector.
//
// Inputs:
//   q:     [1, H, 1, D]  bf16 - query (row vector)
//   k:     [1, H, D, 1]  bf16 - key as COLUMN vector (data in col 0)
//   v:     [1, H, 1, D]  bf16 - pre-computed delta from host
//   decay: [1, H, 1, 1]  bf16 - decay scalar
//   beta:  [1, H, 1, 1]  bf16 - unused
//   state: [1, H, D, D]  fp32 - recurrent state
//
// Kernel:
//   1. state_decayed = state * decay
//   2. state_new = state_decayed + k_col @ delta (outer product)
//   3. output = q @ state_new

#include <cstdint>

#include "api/compute/eltwise_binary.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/bcast.h"
#include "api/compute/matmul.h"
#include "api/compute/pack.h"

constexpr uint32_t cb_q = get_compile_time_arg_val(0);
constexpr uint32_t cb_k = get_compile_time_arg_val(1);
constexpr uint32_t cb_v = get_compile_time_arg_val(2);
constexpr uint32_t cb_decay = get_compile_time_arg_val(3);
constexpr uint32_t cb_beta = get_compile_time_arg_val(4);
constexpr uint32_t cb_state = get_compile_time_arg_val(5);
constexpr uint32_t cb_state_new = get_compile_time_arg_val(6);
constexpr uint32_t cb_out = get_compile_time_arg_val(7);
constexpr uint32_t cb_sd = get_compile_time_arg_val(8);
constexpr uint32_t cb_tmp = get_compile_time_arg_val(9);
constexpr uint32_t cb_tmp2 = get_compile_time_arg_val(10);
constexpr uint32_t cb_sd2 = get_compile_time_arg_val(11);
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

        // Phase 1: fp32 state -> bf16, decay, dual-pack to fp32 + bf16
        cb_reserve_back(cb_sd, STATE_TILES);
        init_sfpu(cb_state, cb_sd);
        for (uint32_t t = 0; t < STATE_TILES; t++) {
            tile_regs_acquire();
            copy_tile(cb_state, t, 0);
            tile_regs_commit();
            tile_regs_wait();
            pack_tile(0, cb_sd);
            tile_regs_release();
        }
        cb_push_back(cb_sd, STATE_TILES);
        cb_pop_front(cb_state, STATE_TILES);

        cb_wait_front(cb_sd, STATE_TILES);
        cb_reserve_back(cb_state_new, STATE_TILES);
        cb_reserve_back(cb_sd2, STATE_TILES);
        binary_op_init_common(cb_sd, cb_decay, cb_state_new);
        mul_tiles_bcast_scalar_init_short(cb_sd, cb_decay);
        for (uint32_t t = 0; t < STATE_TILES; t++) {
            tile_regs_acquire();
            mul_tiles_bcast_scalar(cb_sd, cb_decay, t, 0, 0);
            tile_regs_commit();
            tile_regs_wait();
            pack_tile(0, cb_state_new);
            pack_reconfig_data_format(cb_sd2);
            pack_tile(0, cb_sd2);
            pack_reconfig_data_format(cb_state_new);
            tile_regs_release();
        }
        cb_push_back(cb_state_new, STATE_TILES);
        cb_push_back(cb_sd2, STATE_TILES);
        cb_pop_front(cb_sd, STATE_TILES);
        cb_pop_front(cb_decay, 1);

        // Phase 2: state_new += k_col @ delta (outer product)
        // k_col [D,1]: data in column 0. delta [1,D]: data in row 0.
        // matmul(k_col_tile[i], delta_tile[j]):
        //   C[h,w] = sum_m k_col[h,m]*delta[m,w] = k[h]*delta[w] ✓
        cb_wait_front(cb_sd2, STATE_TILES);
        cb_pop_front(cb_state_new, STATE_TILES);
        cb_reserve_back(cb_state_new, STATE_TILES);

        // Pack outer product to bf16 cb_sd (NOT fp32 cb_state_new, which hangs with mm_init)
        cb_reserve_back(cb_sd, STATE_TILES);
        mm_init(cb_k, cb_v, cb_sd);  // bf16 output works
        for (uint32_t i = 0; i < D_TILES; i++) {
            for (uint32_t j = 0; j < D_TILES; j++) {
                uint32_t sd_idx = i * D_TILES + j;

                tile_regs_acquire();
                matmul_tiles(cb_k, cb_v, i, j, 0);
                tile_regs_commit();
                tile_regs_wait();
                pack_tile(0, cb_sd);
                tile_regs_release();
            }
        }
        cb_push_back(cb_sd, STATE_TILES);
        // Now add: state_new = decayed_state (cb_sd2) + outer_product (cb_sd)
        // Both are bf16. Pack result to fp32 cb_state_new.
        cb_wait_front(cb_sd, STATE_TILES);
        binary_op_init_common(cb_sd2, cb_sd, cb_state_new);
        add_tiles_init(cb_sd2, cb_sd);
        for (uint32_t t = 0; t < STATE_TILES; t++) {
            tile_regs_acquire();
            add_tiles(cb_sd2, cb_sd, t, t, 0);  // bf16+bf16 -> fp32 DST
            tile_regs_commit();
            tile_regs_wait();
            pack_tile(0, cb_state_new);  // fp32 DST -> fp32 CB
            tile_regs_release();
        }
        cb_push_back(cb_state_new, STATE_TILES);
        cb_pop_front(cb_sd, STATE_TILES);
        cb_pop_front(cb_sd2, STATE_TILES);

        // Phase 3: output = q @ state_new[fp32]
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
        cb_pop_front(cb_v, D_TILES);
        cb_pop_front(cb_beta, 1);
    }
}

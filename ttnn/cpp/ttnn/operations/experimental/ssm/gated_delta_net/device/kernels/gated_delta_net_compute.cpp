// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// GatedDeltaNet recurrence: fp32 state, k_col outer product.
//
// CB ownership:
//   cb_state:     compute-internal (reused after reader pops)
//   cb_state_new: compute→writer (pushed ONCE, never popped by compute)
//   cb_sd/cb_sd2: compute-internal
//
// Inputs: q[1,H,1,D], k_col[1,H,D,1], delta[1,H,1,D], decay[1,H,1,1]

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

        // ============================================================
        // Phase 1a: fp32 state → bf16 cb_sd
        // ============================================================
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
        cb_pop_front(cb_state, STATE_TILES);  // cb_state now FREE for reuse

        // ============================================================
        // Phase 1b: decay, dual pack → cb_state(fp32 internal) + cb_sd2(bf16)
        // ============================================================
        cb_wait_front(cb_sd, STATE_TILES);
        cb_reserve_back(cb_state, STATE_TILES);  // REUSE as internal
        cb_reserve_back(cb_sd2, STATE_TILES);
        binary_op_init_common(cb_sd, cb_decay, cb_state);  // packer for fp32
        mul_tiles_bcast_scalar_init_short(cb_sd, cb_decay);
        for (uint32_t t = 0; t < STATE_TILES; t++) {
            tile_regs_acquire();
            mul_tiles_bcast_scalar(cb_sd, cb_decay, t, 0, 0);
            tile_regs_commit();
            tile_regs_wait();
            pack_tile(0, cb_state);  // fp32
            pack_reconfig_data_format(cb_sd2);
            pack_tile(0, cb_sd2);  // bf16
            pack_reconfig_data_format(cb_state);
            tile_regs_release();
        }
        cb_push_back(cb_state, STATE_TILES);  // fp32 decayed (internal)
        cb_push_back(cb_sd2, STATE_TILES);    // bf16 decayed (for add)
        cb_pop_front(cb_sd, STATE_TILES);
        cb_pop_front(cb_decay, 1);

        // ============================================================
        // Phase 2: outer product k_col @ delta → bf16 cb_sd
        // k_col[D,1] col0 data × delta[1,D] row0 data → [D,D] full tile
        // ============================================================
        cb_reserve_back(cb_sd, STATE_TILES);  // reuse cb_sd
        mm_init(cb_k, cb_v, cb_sd);           // bf16 inputs, bf16 output
        for (uint32_t i = 0; i < D_TILES; i++) {
            for (uint32_t j = 0; j < D_TILES; j++) {
                tile_regs_acquire();
                matmul_tiles(cb_k, cb_v, i, j, 0);
                tile_regs_commit();
                tile_regs_wait();
                pack_tile(0, cb_sd);
                tile_regs_release();
            }
        }
        cb_push_back(cb_sd, STATE_TILES);

        // ============================================================
        // Phase 3: state_updated = decayed + outer_product
        // cb_sd2(bf16 decayed) + cb_sd(bf16 outer) → fp32 DST → dual pack
        // → cb_state(fp32 internal, overwrite decayed) + cb_state_new(fp32 output)
        // ============================================================
        cb_wait_front(cb_sd, STATE_TILES);
        cb_wait_front(cb_sd2, STATE_TILES);
        cb_pop_front(cb_state, STATE_TILES);         // free decayed state
        cb_reserve_back(cb_state, STATE_TILES);      // for updated internal
        cb_reserve_back(cb_state_new, STATE_TILES);  // for writer output

        binary_op_init_common(cb_sd2, cb_sd, cb_state);  // packer for fp32
        add_tiles_init(cb_sd2, cb_sd);
        for (uint32_t t = 0; t < STATE_TILES; t++) {
            tile_regs_acquire();
            add_tiles(cb_sd2, cb_sd, t, t, 0);  // bf16+bf16 → fp32 DST
            tile_regs_commit();
            tile_regs_wait();
            pack_tile(0, cb_state);      // fp32 internal (for output matmul)
            pack_tile(0, cb_state_new);  // fp32 output (for writer) -- SAME format, no reconfig
            tile_regs_release();
        }
        cb_push_back(cb_state, STATE_TILES);
        cb_push_back(cb_state_new, STATE_TILES);  // writer's ONLY source, pushed ONCE
        cb_pop_front(cb_sd, STATE_TILES);
        cb_pop_front(cb_sd2, STATE_TILES);

        // ============================================================
        // Phase 4: output = q @ state_updated[fp32]
        // Read from cb_state (internal), NOT cb_state_new (writer's)
        // ============================================================
        cb_wait_front(cb_state, STATE_TILES);
        cb_reserve_back(cb_out, D_TILES);
        mm_init(cb_q, cb_state, cb_out);
        for (uint32_t j = 0; j < D_TILES; j++) {
            tile_regs_acquire();
            for (uint32_t i = 0; i < D_TILES; i++) {
                matmul_tiles(cb_q, cb_state, i, i * D_TILES + j, 0);
            }
            tile_regs_commit();
            tile_regs_wait();
            pack_tile(0, cb_out);
            tile_regs_release();
        }
        cb_push_back(cb_out, D_TILES);
        cb_pop_front(cb_state, STATE_TILES);

        cb_pop_front(cb_q, D_TILES);
        cb_pop_front(cb_k, D_TILES);
        cb_pop_front(cb_v, D_TILES);
        cb_pop_front(cb_beta, 1);
    }
}

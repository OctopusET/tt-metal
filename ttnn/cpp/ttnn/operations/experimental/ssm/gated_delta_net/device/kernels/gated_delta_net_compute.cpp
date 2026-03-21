// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Full GatedDeltaNet recurrence kernel.
// All CBs bf16, fp32_dest_acc_en=true for accumulation precision.
//
// Architecture: 3-phase pipeline per head.
//   Phase 1: state_decayed = state * decay  (bcast scalar mul)
//   Phase 2: kv_mem, delta computation
//   Phase 3: state_new = state_decayed + k^T @ delta, output = q @ state_new
//
// cb_sd is compute-internal: compute produces AND consumes it.
// This avoids the single-producer-single-consumer constraint.

#include <cstdint>

#include "api/compute/eltwise_binary.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/bcast.h"
#include "api/compute/matmul.h"
#include "api/compute/transpose_wh.h"

constexpr uint32_t cb_q = get_compile_time_arg_val(0);
constexpr uint32_t cb_k = get_compile_time_arg_val(1);
constexpr uint32_t cb_v = get_compile_time_arg_val(2);
constexpr uint32_t cb_decay = get_compile_time_arg_val(3);
constexpr uint32_t cb_beta = get_compile_time_arg_val(4);
constexpr uint32_t cb_state = get_compile_time_arg_val(5);
constexpr uint32_t cb_state_new = get_compile_time_arg_val(6);
constexpr uint32_t cb_out = get_compile_time_arg_val(7);
constexpr uint32_t cb_sd = get_compile_time_arg_val(8);     // decayed state (compute internal)
constexpr uint32_t cb_tmp = get_compile_time_arg_val(9);    // scratch for kv_mem
constexpr uint32_t cb_tmp2 = get_compile_time_arg_val(10);  // scratch for delta
constexpr uint32_t D_TILES = get_compile_time_arg_val(11);
constexpr uint32_t STATE_TILES = get_compile_time_arg_val(12);

void kernel_main() {
    const uint32_t num_heads_this_core = get_arg_val<uint32_t>(0);

    for (uint32_t hh = 0; hh < num_heads_this_core; hh++) {
        // Wait for all inputs from reader
        cb_wait_front(cb_q, D_TILES);
        cb_wait_front(cb_k, D_TILES);
        cb_wait_front(cb_v, D_TILES);
        cb_wait_front(cb_decay, 1);
        cb_wait_front(cb_beta, 1);
        cb_wait_front(cb_state, STATE_TILES);

        // ============================================================
        // Phase 1: state_decayed = state * decay (bcast scalar)
        // cb_state -> cb_sd (compute-internal intermediate)
        // ============================================================
        cb_reserve_back(cb_sd, STATE_TILES);
        binary_op_init_common(cb_state, cb_decay, cb_sd);
        mul_tiles_bcast_scalar_init_short(cb_state, cb_decay);
        for (uint32_t t = 0; t < STATE_TILES; t++) {
            tile_regs_acquire();
            mul_tiles_bcast_scalar(cb_state, cb_decay, t, 0, 0);
            tile_regs_commit();
            tile_regs_wait();
            pack_tile(0, cb_sd);
            tile_regs_release();
        }
        cb_push_back(cb_sd, STATE_TILES);
        cb_pop_front(cb_state, STATE_TILES);
        cb_pop_front(cb_decay, 1);

        // ============================================================
        // Phase 2: kv_mem = k @ state_decayed, delta = (v - kv_mem) * beta
        // Read from cb_sd (compute is consumer of its own output)
        // ============================================================
        cb_wait_front(cb_sd, STATE_TILES);

        // kv_mem[j] = sum_i( k_tile[i] @ state_decayed_tile[i*D+j] )
        cb_reserve_back(cb_tmp, D_TILES);
        mm_init(cb_k, cb_sd, cb_tmp);
        for (uint32_t j = 0; j < D_TILES; j++) {
            tile_regs_acquire();
            for (uint32_t i = 0; i < D_TILES; i++) {
                matmul_tiles(cb_k, cb_sd, i, i * D_TILES + j, 0);
            }
            tile_regs_commit();
            tile_regs_wait();
            pack_tile(0, cb_tmp);
            tile_regs_release();
        }
        cb_push_back(cb_tmp, D_TILES);  // cb_tmp has kv_mem

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

        // delta *= beta (scalar broadcast)
        cb_wait_front(cb_tmp2, D_TILES);
        cb_reserve_back(cb_tmp, D_TILES);  // reuse cb_tmp for final delta
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
        cb_push_back(cb_tmp, D_TILES);  // cb_tmp has delta
        cb_pop_front(cb_tmp2, D_TILES);
        cb_pop_front(cb_beta, 1);

        // ============================================================
        // Phase 3: state_new = state_decayed + k^T @ delta
        //          output = q @ state_new
        //
        // Step 1: Transpose k tiles within each tile (row0 -> col0)
        // Step 2: For each (i,j): update = k_t[i] @ delta[j], state_new = sd + update
        // ============================================================
        cb_wait_front(cb_tmp, D_TILES);  // delta

        // Transpose k tiles: row-oriented -> column-oriented within each tile
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
        cb_push_back(cb_tmp2, D_TILES);  // cb_tmp2 has k_transposed
        cb_wait_front(cb_tmp2, D_TILES);

        // Compute state_new[i,j] = sd[i,j] + k_t[i] @ delta[j]
        cb_reserve_back(cb_state_new, STATE_TILES);

        for (uint32_t i = 0; i < D_TILES; i++) {
            for (uint32_t j = 0; j < D_TILES; j++) {
                uint32_t sd_idx = i * D_TILES + j;

                // Outer product tile: k_t_tile[i] @ delta_tile[j]
                mm_init(cb_tmp2, cb_tmp, cb_state_new);
                tile_regs_acquire();
                matmul_tiles(cb_tmp2, cb_tmp, i, j, 0);
                tile_regs_commit();
                tile_regs_wait();

                // Add state_decayed: state_new[i,j] = dst[0] + sd[i,j]
                // Use binary_dest_reuse: dst[0] += cb_sd[sd_idx]
                binary_dest_reuse_tiles_init(cb_sd);
                binary_dest_reuse_tiles(cb_sd, sd_idx, 0);

                pack_tile(0, cb_state_new);
                tile_regs_release();
            }
        }
        cb_push_back(cb_state_new, STATE_TILES);
        cb_pop_front(cb_sd, STATE_TILES);  // done with decayed state
        cb_pop_front(cb_tmp, D_TILES);     // done with delta
        cb_pop_front(cb_tmp2, D_TILES);    // done with k_transposed

        // Output: q @ state_new
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

        // Pop inputs (writer will pop state_new and out)
        cb_pop_front(cb_q, D_TILES);
        cb_pop_front(cb_k, D_TILES);
    }
}

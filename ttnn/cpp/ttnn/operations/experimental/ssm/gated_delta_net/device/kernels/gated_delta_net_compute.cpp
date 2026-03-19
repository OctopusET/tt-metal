// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Fused GatedDeltaNet recurrence compute kernel.
//
// Per head, all in fp32 (fp32_dest_acc_en=true):
//   1. state *= decay                     (scalar broadcast multiply)
//   2. kv_mem = k @ state                 (matmul with accumulation)
//   3. delta = (v - kv_mem) * beta        (sub + scalar mul)
//   4. state += k^T @ delta               (transpose + matmul outer product)
//   5. output = q @ state                 (matmul with accumulation)

#include <cstdint>

#include "api/compute/eltwise_binary.h"
#include "api/compute/matmul.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/transpose_wh.h"
#include "api/compute/bcast.h"

constexpr uint32_t cb_q = get_compile_time_arg_val(0);
constexpr uint32_t cb_k = get_compile_time_arg_val(1);
constexpr uint32_t cb_v = get_compile_time_arg_val(2);
constexpr uint32_t cb_decay = get_compile_time_arg_val(3);
constexpr uint32_t cb_beta = get_compile_time_arg_val(4);
constexpr uint32_t cb_state = get_compile_time_arg_val(5);
constexpr uint32_t cb_state_new = get_compile_time_arg_val(6);
constexpr uint32_t cb_out = get_compile_time_arg_val(7);
constexpr uint32_t cb_kv_mem = get_compile_time_arg_val(8);
constexpr uint32_t cb_delta = get_compile_time_arg_val(9);
constexpr uint32_t cb_k_t = get_compile_time_arg_val(10);
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

        // ===============================================================
        // Step 1: Decay -- state_new = state * decay (scalar broadcast)
        // ===============================================================
        cb_reserve_back(cb_state_new, STATE_TILES);
        mul_tiles_bcast_scalar_init_short(cb_state, cb_decay);
        for (uint32_t t = 0; t < STATE_TILES; t++) {
            tile_regs_acquire();
            mul_tiles_bcast_scalar(cb_state, cb_decay, t, 0, 0);
            tile_regs_commit();
            tile_regs_wait();
            pack_tile(0, cb_state_new);
            tile_regs_release();
        }
        cb_push_back(cb_state_new, STATE_TILES);
        cb_pop_front(cb_state, STATE_TILES);
        cb_pop_front(cb_decay, 1);

        // ===============================================================
        // Step 2: kv_mem = k @ state_new
        // kv_mem[c] = sum_{r=0}^{D_TILES-1} k[r] @ state[r*D_TILES+c]
        // Result: D_TILES bf16 tiles in cb_kv_mem
        // ===============================================================
        cb_wait_front(cb_state_new, STATE_TILES);
        cb_reserve_back(cb_kv_mem, D_TILES);
        mm_init_short(cb_k, cb_state_new);
        for (uint32_t c = 0; c < D_TILES; c++) {
            tile_regs_acquire();
            for (uint32_t r = 0; r < D_TILES; r++) {
                matmul_tiles(cb_k, cb_state_new, r, r * D_TILES + c, 0);
            }
            tile_regs_commit();
            tile_regs_wait();
            pack_tile(0, cb_kv_mem);
            tile_regs_release();
        }
        cb_push_back(cb_kv_mem, D_TILES);

        // ===============================================================
        // Step 3: delta = (v - kv_mem) * beta
        // ===============================================================
        cb_wait_front(cb_kv_mem, D_TILES);
        cb_reserve_back(cb_delta, D_TILES);
        sub_tiles_init(cb_v, cb_kv_mem);
        for (uint32_t c = 0; c < D_TILES; c++) {
            tile_regs_acquire();
            sub_tiles(cb_v, cb_kv_mem, c, c, 0);
            tile_regs_commit();
            tile_regs_wait();
            pack_tile(0, cb_delta);
            tile_regs_release();
        }
        cb_push_back(cb_delta, D_TILES);
        cb_pop_front(cb_v, D_TILES);
        cb_pop_front(cb_kv_mem, D_TILES);

        // Multiply delta by beta (scalar broadcast, in-place via pop/push)
        cb_wait_front(cb_delta, D_TILES);
        mul_tiles_bcast_scalar_init_short(cb_delta, cb_beta);
        for (uint32_t c = 0; c < D_TILES; c++) {
            tile_regs_acquire();
            mul_tiles_bcast_scalar(cb_delta, cb_beta, c, 0, 0);
            tile_regs_commit();
            tile_regs_wait();
            // Overwrite in place: pack to same position
            pack_tile(0, cb_delta);
            tile_regs_release();
        }
        // Pop old, reserve+push new (in-place pattern)
        cb_pop_front(cb_delta, D_TILES);
        cb_reserve_back(cb_delta, D_TILES);
        cb_push_back(cb_delta, D_TILES);
        cb_pop_front(cb_beta, 1);

        // ===============================================================
        // Step 4: Transpose k, then state += outer(k^T, delta)
        // ===============================================================
        // Transpose k: row-0 data -> column-0 data
        cb_reserve_back(cb_k_t, D_TILES);
        transpose_wh_init(cb_k, cb_k_t);
        for (uint32_t r = 0; r < D_TILES; r++) {
            tile_regs_acquire();
            transpose_wh_tile(cb_k, r, 0);
            tile_regs_commit();
            tile_regs_wait();
            pack_tile(0, cb_k_t);
            tile_regs_release();
        }
        cb_push_back(cb_k_t, D_TILES);
        cb_pop_front(cb_k, D_TILES);

        // State update: for each tile [r][c], add outer product k_t[r] @ delta[c]
        // state_new is currently the decayed state. We read it, add outer product,
        // write to cb_state (reusing as output buffer).
        cb_wait_front(cb_k_t, D_TILES);
        cb_wait_front(cb_delta, D_TILES);
        cb_reserve_back(cb_state, STATE_TILES);

        for (uint32_t r = 0; r < D_TILES; r++) {
            for (uint32_t c = 0; c < D_TILES; c++) {
                uint32_t state_idx = r * D_TILES + c;

                // Copy decayed state tile into DST[0]
                copy_tile_to_dst_init_short(cb_state_new);
                tile_regs_acquire();
                copy_tile(cb_state_new, state_idx, 0);

                // Add outer product: DST[0] += k_t[r] @ delta[c]
                mm_init_short(cb_k_t, cb_delta);
                matmul_tiles(cb_k_t, cb_delta, r, c, 0);

                tile_regs_commit();
                tile_regs_wait();
                pack_tile(0, cb_state);
                tile_regs_release();
            }
        }
        cb_push_back(cb_state, STATE_TILES);
        cb_pop_front(cb_state_new, STATE_TILES);
        cb_pop_front(cb_k_t, D_TILES);
        cb_pop_front(cb_delta, D_TILES);

        // ===============================================================
        // Step 5: output = q @ state_updated
        // output[c] = sum_{r=0}^{D_TILES-1} q[r] @ state[r*D_TILES+c]
        // ===============================================================
        cb_wait_front(cb_state, STATE_TILES);
        cb_reserve_back(cb_out, D_TILES);
        mm_init_short(cb_q, cb_state);
        for (uint32_t c = 0; c < D_TILES; c++) {
            tile_regs_acquire();
            for (uint32_t r = 0; r < D_TILES; r++) {
                matmul_tiles(cb_q, cb_state, r, r * D_TILES + c, 0);
            }
            tile_regs_commit();
            tile_regs_wait();
            pack_tile(0, cb_out);
            tile_regs_release();
        }
        cb_push_back(cb_out, D_TILES);
        cb_pop_front(cb_q, D_TILES);

        // State remains in cb_state for writer to flush.
        // Writer reads cb_state as cb_state_new (step 4 wrote updated state there).
        // NOTE: The writer uses cb_state_new, but we wrote the final state to cb_state.
        // We need to move it. Simplest: rename in program factory so writer reads cb_state.
        // For now, copy state to cb_state_new for the writer.
        cb_reserve_back(cb_state_new, STATE_TILES);
        copy_tile_to_dst_init_short(cb_state);
        for (uint32_t t = 0; t < STATE_TILES; t++) {
            tile_regs_acquire();
            copy_tile(cb_state, t, 0);
            tile_regs_commit();
            tile_regs_wait();
            pack_tile(0, cb_state_new);
            tile_regs_release();
        }
        cb_push_back(cb_state_new, STATE_TILES);
        cb_pop_front(cb_state, STATE_TILES);
    }
}

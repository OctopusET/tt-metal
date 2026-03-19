// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Simplified GatedDeltaNet compute kernel: decay only (for debugging CB flow).
// Full recurrence to be added after infrastructure works.

#include <cstdint>

#include "api/compute/eltwise_binary.h"
#include "api/compute/tile_move_copy.h"
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
        // Wait for all inputs
        cb_wait_front(cb_q, D_TILES);
        cb_wait_front(cb_k, D_TILES);
        cb_wait_front(cb_v, D_TILES);
        cb_wait_front(cb_decay, 1);
        cb_wait_front(cb_beta, 1);
        cb_wait_front(cb_state, STATE_TILES);

        // Step 1: Decay -- state_new = state * decay (placeholder: just copy state)
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

        // Output: just copy q as placeholder
        cb_reserve_back(cb_out, D_TILES);
        copy_tile_to_dst_init_short(cb_q);
        for (uint32_t t = 0; t < D_TILES; t++) {
            tile_regs_acquire();
            copy_tile(cb_q, t, 0);
            tile_regs_commit();
            tile_regs_wait();
            pack_tile(0, cb_out);
            tile_regs_release();
        }
        cb_push_back(cb_out, D_TILES);

        // Pop remaining inputs
        cb_pop_front(cb_q, D_TILES);
        cb_pop_front(cb_k, D_TILES);
        cb_pop_front(cb_v, D_TILES);
        cb_pop_front(cb_decay, 1);
        cb_pop_front(cb_beta, 1);
    }
}

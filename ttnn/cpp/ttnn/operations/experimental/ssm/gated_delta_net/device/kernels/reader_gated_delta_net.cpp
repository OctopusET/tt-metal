// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Reader kernel for GatedDeltaNet: reads q, k, v, decay, beta, state from DRAM.
// Each core handles one head. Tiles are read using noc_async_read_tile.

#include <cstdint>
#include "api/dataflow/dataflow_api.h"

constexpr uint32_t cb_q = get_compile_time_arg_val(0);
constexpr uint32_t cb_k = get_compile_time_arg_val(1);
constexpr uint32_t cb_v = get_compile_time_arg_val(2);
constexpr uint32_t cb_decay = get_compile_time_arg_val(3);
constexpr uint32_t cb_beta = get_compile_time_arg_val(4);
constexpr uint32_t cb_state = get_compile_time_arg_val(5);
constexpr uint32_t D_TILES = get_compile_time_arg_val(6);
constexpr uint32_t STATE_TILES = get_compile_time_arg_val(7);

void kernel_main() {
    uint32_t q_addr = get_arg_val<uint32_t>(0);
    uint32_t k_addr = get_arg_val<uint32_t>(1);
    uint32_t v_addr = get_arg_val<uint32_t>(2);
    uint32_t decay_addr = get_arg_val<uint32_t>(3);
    uint32_t beta_addr = get_arg_val<uint32_t>(4);
    uint32_t state_addr = get_arg_val<uint32_t>(5);
    uint32_t head_start = get_arg_val<uint32_t>(6);
    uint32_t num_heads_this_core = get_arg_val<uint32_t>(7);
    uint32_t bf16_tile_bytes = get_arg_val<uint32_t>(8);
    uint32_t fp32_tile_bytes = get_arg_val<uint32_t>(9);

    // For interleaved tile layout, tile index = head * tiles_per_head + offset
    // q,k,v: (1, H, 1, D) -> H * D_TILES tiles, head h starts at h * D_TILES
    // decay,beta: (1, H, 1, 1) -> H tiles, head h is tile h
    // state: (1, H, D, D) -> H * STATE_TILES tiles, head h starts at h * STATE_TILES

    for (uint32_t hh = 0; hh < num_heads_this_core; hh++) {
        uint32_t h = head_start + hh;

        // Read q vector (D_TILES bf16 tiles)
        uint32_t q_tile_start = h * D_TILES;
        for (uint32_t t = 0; t < D_TILES; t++) {
            cb_reserve_back(cb_q, 1);
            uint64_t q_noc_addr = get_noc_addr(q_tile_start + t, q_addr, bf16_tile_bytes);
            noc_async_read(q_noc_addr, get_write_ptr(cb_q), bf16_tile_bytes);
            noc_async_read_barrier();
            cb_push_back(cb_q, 1);
        }

        // Read k vector
        uint32_t k_tile_start = h * D_TILES;
        for (uint32_t t = 0; t < D_TILES; t++) {
            cb_reserve_back(cb_k, 1);
            uint64_t k_noc_addr = get_noc_addr(k_tile_start + t, k_addr, bf16_tile_bytes);
            noc_async_read(k_noc_addr, get_write_ptr(cb_k), bf16_tile_bytes);
            noc_async_read_barrier();
            cb_push_back(cb_k, 1);
        }

        // Read v vector
        uint32_t v_tile_start = h * D_TILES;
        for (uint32_t t = 0; t < D_TILES; t++) {
            cb_reserve_back(cb_v, 1);
            uint64_t v_noc_addr = get_noc_addr(v_tile_start + t, v_addr, bf16_tile_bytes);
            noc_async_read(v_noc_addr, get_write_ptr(cb_v), bf16_tile_bytes);
            noc_async_read_barrier();
            cb_push_back(cb_v, 1);
        }

        // Read decay (1 tile)
        cb_reserve_back(cb_decay, 1);
        uint64_t decay_noc_addr = get_noc_addr(h, decay_addr, bf16_tile_bytes);
        noc_async_read(decay_noc_addr, get_write_ptr(cb_decay), bf16_tile_bytes);
        noc_async_read_barrier();
        cb_push_back(cb_decay, 1);

        // Read beta (1 tile)
        cb_reserve_back(cb_beta, 1);
        uint64_t beta_noc_addr = get_noc_addr(h, beta_addr, bf16_tile_bytes);
        noc_async_read(beta_noc_addr, get_write_ptr(cb_beta), bf16_tile_bytes);
        noc_async_read_barrier();
        cb_push_back(cb_beta, 1);

        // Read state (STATE_TILES fp32 tiles)
        uint32_t state_tile_start = h * STATE_TILES;
        for (uint32_t t = 0; t < STATE_TILES; t++) {
            cb_reserve_back(cb_state, 1);
            uint64_t s_noc_addr = get_noc_addr(state_tile_start + t, state_addr, fp32_tile_bytes);
            noc_async_read(s_noc_addr, get_write_ptr(cb_state), fp32_tile_bytes);
            noc_async_read_barrier();
            cb_push_back(cb_state, 1);
        }
    }
}

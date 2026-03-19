// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Writer kernel for GatedDeltaNet: writes output vector and updated state to DRAM.

#include <cstdint>
#include "api/dataflow/dataflow_api.h"

constexpr uint32_t cb_out = get_compile_time_arg_val(0);
constexpr uint32_t cb_state_new = get_compile_time_arg_val(1);
constexpr uint32_t D_TILES = get_compile_time_arg_val(2);
constexpr uint32_t STATE_TILES = get_compile_time_arg_val(3);

void kernel_main() {
    uint32_t out_addr = get_arg_val<uint32_t>(0);
    uint32_t state_addr = get_arg_val<uint32_t>(1);
    uint32_t head_start = get_arg_val<uint32_t>(2);
    uint32_t num_heads_this_core = get_arg_val<uint32_t>(3);
    uint32_t bf16_tile_bytes = get_arg_val<uint32_t>(4);
    uint32_t fp32_tile_bytes = get_arg_val<uint32_t>(5);

    for (uint32_t hh = 0; hh < num_heads_this_core; hh++) {
        uint32_t h = head_start + hh;

        // Write output vector (D_TILES bf16 tiles)
        uint32_t out_tile_start = h * D_TILES;
        cb_wait_front(cb_out, D_TILES);
        uint32_t out_l1_addr = get_read_ptr(cb_out);
        for (uint32_t t = 0; t < D_TILES; t++) {
            uint64_t out_noc_addr = get_noc_addr(out_tile_start + t, out_addr, bf16_tile_bytes);
            noc_async_write(out_l1_addr, out_noc_addr, bf16_tile_bytes);
            out_l1_addr += bf16_tile_bytes;
        }
        noc_async_write_barrier();
        cb_pop_front(cb_out, D_TILES);

        // Write updated state (STATE_TILES fp32 tiles)
        uint32_t state_tile_start = h * STATE_TILES;
        cb_wait_front(cb_state_new, STATE_TILES);
        uint32_t state_l1_addr = get_read_ptr(cb_state_new);
        for (uint32_t t = 0; t < STATE_TILES; t++) {
            uint64_t s_noc_addr = get_noc_addr(state_tile_start + t, state_addr, fp32_tile_bytes);
            noc_async_write(state_l1_addr, s_noc_addr, fp32_tile_bytes);
            state_l1_addr += fp32_tile_bytes;
        }
        noc_async_write_barrier();
        cb_pop_front(cb_state_new, STATE_TILES);
    }
}

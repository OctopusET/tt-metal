// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"

constexpr uint32_t cb_out = get_compile_time_arg_val(0);
constexpr uint32_t cb_state_new = get_compile_time_arg_val(1);
constexpr uint32_t D_TILES = get_compile_time_arg_val(2);
constexpr uint32_t STATE_TILES = get_compile_time_arg_val(3);

// TensorAccessor args start at compile-time arg 4
constexpr auto out_acc_args = TensorAccessorArgs<4>();
constexpr auto state_acc_args = TensorAccessorArgs<out_acc_args.next_compile_time_args_offset()>();

void kernel_main() {
    uint32_t out_addr = get_arg_val<uint32_t>(0);
    uint32_t state_addr = get_arg_val<uint32_t>(1);
    uint32_t head_start = get_arg_val<uint32_t>(2);
    uint32_t num_heads = get_arg_val<uint32_t>(3);

    const uint32_t bf16_tile_bytes = get_tile_size(cb_out);
    const uint32_t fp32_tile_bytes = get_tile_size(cb_state_new);

    const auto out_acc = TensorAccessor(out_acc_args, out_addr, bf16_tile_bytes);
    const auto state_acc = TensorAccessor(state_acc_args, state_addr, fp32_tile_bytes);

    for (uint32_t hh = 0; hh < num_heads; hh++) {
        uint32_t h = head_start + hh;

        // Write updated state FIRST (compute produces this first)
        cb_wait_front(cb_state_new, STATE_TILES);
        uint32_t state_l1 = get_read_ptr(cb_state_new);
        for (uint32_t t = 0; t < STATE_TILES; t++) {
            noc_async_write_tile(h * STATE_TILES + t, state_acc, state_l1);
            state_l1 += fp32_tile_bytes;
        }
        noc_async_write_barrier();
        cb_pop_front(cb_state_new, STATE_TILES);

        // Write output SECOND (compute produces this second)
        cb_wait_front(cb_out, D_TILES);
        uint32_t out_l1 = get_read_ptr(cb_out);
        for (uint32_t t = 0; t < D_TILES; t++) {
            noc_async_write_tile(h * D_TILES + t, out_acc, out_l1);
            out_l1 += bf16_tile_bytes;
        }
        noc_async_write_barrier();
        cb_pop_front(cb_out, D_TILES);
    }
}

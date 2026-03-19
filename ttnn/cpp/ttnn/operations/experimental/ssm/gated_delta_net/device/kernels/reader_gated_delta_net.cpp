// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"

constexpr uint32_t cb_q = get_compile_time_arg_val(0);
constexpr uint32_t cb_k = get_compile_time_arg_val(1);
constexpr uint32_t cb_v = get_compile_time_arg_val(2);
constexpr uint32_t cb_decay = get_compile_time_arg_val(3);
constexpr uint32_t cb_beta = get_compile_time_arg_val(4);
constexpr uint32_t cb_state = get_compile_time_arg_val(5);
constexpr uint32_t D_TILES = get_compile_time_arg_val(6);
constexpr uint32_t STATE_TILES = get_compile_time_arg_val(7);

// TensorAccessor args start at compile-time arg 8
constexpr auto q_acc_args = TensorAccessorArgs<8>();
constexpr auto k_acc_args = TensorAccessorArgs<q_acc_args.next_compile_time_args_offset()>();
constexpr auto v_acc_args = TensorAccessorArgs<k_acc_args.next_compile_time_args_offset()>();
constexpr auto decay_acc_args = TensorAccessorArgs<v_acc_args.next_compile_time_args_offset()>();
constexpr auto beta_acc_args = TensorAccessorArgs<decay_acc_args.next_compile_time_args_offset()>();
constexpr auto state_acc_args = TensorAccessorArgs<beta_acc_args.next_compile_time_args_offset()>();

void kernel_main() {
    uint32_t q_addr = get_arg_val<uint32_t>(0);
    uint32_t k_addr = get_arg_val<uint32_t>(1);
    uint32_t v_addr = get_arg_val<uint32_t>(2);
    uint32_t decay_addr = get_arg_val<uint32_t>(3);
    uint32_t beta_addr = get_arg_val<uint32_t>(4);
    uint32_t state_addr = get_arg_val<uint32_t>(5);
    uint32_t head_start = get_arg_val<uint32_t>(6);
    uint32_t num_heads = get_arg_val<uint32_t>(7);

    const uint32_t bf16_tile_bytes = get_tile_size(cb_q);
    const uint32_t fp32_tile_bytes = get_tile_size(cb_state);

    const auto q_acc = TensorAccessor(q_acc_args, q_addr, bf16_tile_bytes);
    const auto k_acc = TensorAccessor(k_acc_args, k_addr, bf16_tile_bytes);
    const auto v_acc = TensorAccessor(v_acc_args, v_addr, bf16_tile_bytes);
    const auto decay_acc = TensorAccessor(decay_acc_args, decay_addr, bf16_tile_bytes);
    const auto beta_acc = TensorAccessor(beta_acc_args, beta_addr, bf16_tile_bytes);
    const auto state_acc = TensorAccessor(state_acc_args, state_addr, fp32_tile_bytes);

    for (uint32_t hh = 0; hh < num_heads; hh++) {
        uint32_t h = head_start + hh;

        // Read q (D_TILES bf16 tiles)
        for (uint32_t t = 0; t < D_TILES; t++) {
            cb_reserve_back(cb_q, 1);
            noc_async_read_tile(h * D_TILES + t, q_acc, get_write_ptr(cb_q));
            noc_async_read_barrier();
            cb_push_back(cb_q, 1);
        }

        // Read k
        for (uint32_t t = 0; t < D_TILES; t++) {
            cb_reserve_back(cb_k, 1);
            noc_async_read_tile(h * D_TILES + t, k_acc, get_write_ptr(cb_k));
            noc_async_read_barrier();
            cb_push_back(cb_k, 1);
        }

        // Read v
        for (uint32_t t = 0; t < D_TILES; t++) {
            cb_reserve_back(cb_v, 1);
            noc_async_read_tile(h * D_TILES + t, v_acc, get_write_ptr(cb_v));
            noc_async_read_barrier();
            cb_push_back(cb_v, 1);
        }

        // Read decay (1 tile)
        cb_reserve_back(cb_decay, 1);
        noc_async_read_tile(h, decay_acc, get_write_ptr(cb_decay));
        noc_async_read_barrier();
        cb_push_back(cb_decay, 1);

        // Read beta (1 tile)
        cb_reserve_back(cb_beta, 1);
        noc_async_read_tile(h, beta_acc, get_write_ptr(cb_beta));
        noc_async_read_barrier();
        cb_push_back(cb_beta, 1);

        // Read state (STATE_TILES fp32 tiles)
        for (uint32_t t = 0; t < STATE_TILES; t++) {
            cb_reserve_back(cb_state, 1);
            noc_async_read_tile(h * STATE_TILES + t, state_acc, get_write_ptr(cb_state));
            noc_async_read_barrier();
            cb_push_back(cb_state, 1);
        }
    }
}

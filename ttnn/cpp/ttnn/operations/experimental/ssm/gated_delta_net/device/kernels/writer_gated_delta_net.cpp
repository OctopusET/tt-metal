// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Writer for fused GatedDeltaNet kernel with batch support.
// For each head, collects BATCH_SIZE outputs: copies row 0 of each into row b of accumulator tile.
// Writes per-batch state to batch-offset DRAM position.

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"

constexpr uint32_t cb_out = get_compile_time_arg_val(0);
constexpr uint32_t cb_state_new = get_compile_time_arg_val(1);
constexpr uint32_t D_TILES = get_compile_time_arg_val(2);
constexpr uint32_t STATE_TILES = get_compile_time_arg_val(3);
constexpr uint32_t BATCH_SIZE = get_compile_time_arg_val(4);
constexpr uint32_t STATE_BATCH_STRIDE = get_compile_time_arg_val(5);
constexpr uint32_t cb_out_accum = get_compile_time_arg_val(6);

// TensorAccessor args start after compile-time arg 7
constexpr auto out_acc_args = TensorAccessorArgs<7>();
constexpr auto state_acc_args = TensorAccessorArgs<out_acc_args.next_compile_time_args_offset()>();

void kernel_main() {
    uint32_t out_addr = get_arg_val<uint32_t>(0);
    uint32_t state_addr = get_arg_val<uint32_t>(1);
    uint32_t head_start = get_arg_val<uint32_t>(2);
    uint32_t num_heads = get_arg_val<uint32_t>(3);

    const uint32_t bf16_tile_bytes = get_tile_size(cb_out);
    const uint32_t state_tile_bytes = get_tile_size(cb_state_new);

    const auto out_acc = TensorAccessor(out_acc_args, out_addr, bf16_tile_bytes);
    const auto state_acc = TensorAccessor(state_acc_args, state_addr, state_tile_bytes);

    for (uint32_t hh = 0; hh < num_heads; hh++) {
        uint32_t h = head_start + hh;

        // Allocate and zero-fill output accumulator tiles in L1
        cb_reserve_back(cb_out_accum, D_TILES);
        uint32_t accum_base = get_write_ptr(cb_out_accum);
        {
            // Zero-fill via L1 store (not NOC read, which can hang on writer/NCRISC)
            volatile tt_l1_ptr uint32_t* zptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(accum_base);
            for (uint32_t i = 0; i < D_TILES * (bf16_tile_bytes / 4); i++) {
                zptr[i] = 0;
            }
        }

        for (uint32_t bi = 0; bi < BATCH_SIZE; bi++) {
            // Write updated state for this batch item FIRST
            cb_wait_front(cb_state_new, STATE_TILES);
            uint32_t state_l1 = get_read_ptr(cb_state_new);
            for (uint32_t t = 0; t < STATE_TILES; t++) {
                noc_async_write_tile(bi * STATE_BATCH_STRIDE + h * STATE_TILES + t, state_acc, state_l1);
                state_l1 += state_tile_bytes;
            }
            noc_async_write_barrier();
            cb_pop_front(cb_state_new, STATE_TILES);

            // Copy row 0 of output to row bi of accumulator SECOND
            cb_wait_front(cb_out, D_TILES);
            uint32_t out_l1 = get_read_ptr(cb_out);

            // Destination row offsets for bi (uint16_t units)
            uint32_t dst_off0, dst_off1;
            if (bi < 16) {
                dst_off0 = bi * 16;
                dst_off1 = 256 + bi * 16;
            } else {
                dst_off0 = 512 + (bi - 16) * 16;
                dst_off1 = 768 + (bi - 16) * 16;
            }

            for (uint32_t t = 0; t < D_TILES; t++) {
                volatile tt_l1_ptr uint16_t* src =
                    reinterpret_cast<volatile tt_l1_ptr uint16_t*>(out_l1 + t * bf16_tile_bytes);
                volatile tt_l1_ptr uint16_t* dst =
                    reinterpret_cast<volatile tt_l1_ptr uint16_t*>(accum_base + t * bf16_tile_bytes);
                // Row 0 of src (face 0: 0-15, face 1: 256-271) → row bi of dst
                for (uint32_t i = 0; i < 16; i++) {
                    dst[dst_off0 + i] = src[i];
                    dst[dst_off1 + i] = src[256 + i];
                }
            }
            cb_pop_front(cb_out, D_TILES);
        }

        // Write assembled output tiles to DRAM
        cb_push_back(cb_out_accum, D_TILES);
        uint32_t accum_read = get_read_ptr(cb_out_accum);
        for (uint32_t t = 0; t < D_TILES; t++) {
            noc_async_write_tile(h * D_TILES + t, out_acc, accum_read + t * bf16_tile_bytes);
        }
        noc_async_write_barrier();
        cb_pop_front(cb_out_accum, D_TILES);
    }
}

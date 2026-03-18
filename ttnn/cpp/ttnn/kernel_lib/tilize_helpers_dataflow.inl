// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

#pragma once

/**
 * @file tilize_helpers_dataflow.inl
 * @brief Implementation of tilize/untilize dataflow helpers
 *
 * This file contains the implementation details for read_sticks_for_tilize()
 * and write_sticks_after_untilize(). It should only be included by
 * tilize_helpers_dataflow.hpp.
 */

namespace dataflow_kernel_lib {

namespace detail {

constexpr uint32_t round_up(uint32_t value, uint32_t multiple) {
    return ((value + multiple - 1) / multiple) * multiple;
}

constexpr uint32_t div_up(uint32_t a, uint32_t b) {
    return (a + b - 1) / b;
}

}  // namespace detail

template <uint32_t cb_id, TilizeGranularity granularity, typename Accessor>
FORCE_INLINE void read_sticks_for_tilize(const Accessor& accessor, uint32_t total_num_rows, uint32_t row_bytes) {
    // Derive tile geometry from CB configuration (all constexpr)
    constexpr uint32_t tile_h = unpack_tile_r_dim[cb_id];
    constexpr uint32_t tile_w = unpack_tile_c_dim[cb_id];
    constexpr uint32_t tile_size = get_tile_size(cb_id);
    constexpr uint32_t tile_hw = get_tile_hw(cb_id);
    constexpr uint32_t elem_size = tile_size / tile_hw;
    constexpr uint32_t tile_row_bytes = tile_w * elem_size;

    // elem_size derivation only valid for standard (non block-float) formats
    ASSERT(tile_size % tile_hw == 0);

    // Pad row width up to tile boundary
    uint32_t padded_row_bytes = detail::round_up(row_bytes, tile_row_bytes);
    uint32_t width_in_tiles = padded_row_bytes / tile_row_bytes;
    uint32_t total_blocks = detail::div_up(total_num_rows, tile_h);

    if constexpr (granularity == TilizeGranularity::TILE) {
        // TILE mode: push width_in_tiles pages per block
        // CB page_size must be tile_size
        uint32_t cb_capacity = get_local_cb_interface(cb_id).fifo_num_pages;
        if (cb_capacity > 0) {
            ASSERT(width_in_tiles <= cb_capacity);
        }

        for (uint32_t block = 0; block < total_blocks; block++) {
            uint32_t start_row = block * tile_h;
            uint32_t rows_this_block = total_num_rows - start_row;
            if (rows_this_block > tile_h) {
                rows_this_block = tile_h;
            }

            cb_reserve_back(cb_id, width_in_tiles);
            uint32_t l1_addr = get_write_ptr(cb_id);

            for (uint32_t row = 0; row < rows_this_block; row++) {
                uint64_t noc_addr = accessor.get_noc_addr(start_row + row);
                noc_async_read(noc_addr, l1_addr, row_bytes);
                l1_addr += padded_row_bytes;
            }

            noc_async_read_barrier();
            cb_push_back(cb_id, width_in_tiles);
        }
    } else {
        // ROW mode: push 1 page per row
        // CB page_size must be padded_row_bytes
        for (uint32_t row = 0; row < total_num_rows; row++) {
            cb_reserve_back(cb_id, 1);
            uint32_t l1_addr = get_write_ptr(cb_id);

            uint64_t noc_addr = accessor.get_noc_addr(row);
            noc_async_read(noc_addr, l1_addr, row_bytes);

            noc_async_read_barrier();
            cb_push_back(cb_id, 1);
        }
    }
}

template <uint32_t cb_id, typename Accessor>
FORCE_INLINE void write_sticks_after_untilize(const Accessor& accessor, uint32_t total_num_rows, uint32_t row_bytes) {
    // Derive tile geometry from CB configuration (all constexpr)
    constexpr uint32_t tile_h = unpack_tile_r_dim[cb_id];
    constexpr uint32_t tile_w = unpack_tile_c_dim[cb_id];
    constexpr uint32_t tile_size = get_tile_size(cb_id);
    constexpr uint32_t tile_hw = get_tile_hw(cb_id);
    constexpr uint32_t elem_size = tile_size / tile_hw;
    constexpr uint32_t tile_row_bytes = tile_w * elem_size;

    ASSERT(tile_size % tile_hw == 0);

    // Pad row width up to tile boundary
    uint32_t padded_row_bytes = detail::round_up(row_bytes, tile_row_bytes);
    uint32_t width_in_tiles = padded_row_bytes / tile_row_bytes;
    uint32_t total_blocks = detail::div_up(total_num_rows, tile_h);

    for (uint32_t block = 0; block < total_blocks; block++) {
        uint32_t start_row = block * tile_h;
        uint32_t rows_this_block = total_num_rows - start_row;
        if (rows_this_block > tile_h) {
            rows_this_block = tile_h;
        }

        // Wait for full tile pages from compute (untilize produces full tiles)
        cb_wait_front(cb_id, width_in_tiles);
        uint32_t l1_addr = get_read_ptr(cb_id);

        for (uint32_t row = 0; row < rows_this_block; row++) {
            uint64_t noc_addr = accessor.get_noc_addr(start_row + row);
            noc_async_write(l1_addr, noc_addr, row_bytes);
            l1_addr += padded_row_bytes;
        }

        noc_async_write_barrier();
        cb_pop_front(cb_id, width_in_tiles);
    }
}

}  // namespace dataflow_kernel_lib

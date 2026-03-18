// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "api/dataflow/dataflow_api.h"

namespace dataflow_kernel_lib {

/**
 * @brief Controls CB page granularity for tilize dataflow
 *
 * TILE: CB page_size = tile_size. Reader pushes width_in_tiles pages per block.
 *       Compute tilize uses symmetric mode. Coarser, less synchronization.
 *
 * ROW:  CB page_size = padded_row_bytes. Reader pushes 1 page per row.
 *       Compute tilize uses asymmetric mode (total_input_pages). Finer
 *       granularity — compute can start as soon as 32 rows arrive.
 */
enum class TilizeGranularity : uint8_t {
    TILE,
    ROW,
};

/**
 * @brief Read row-major sticks from DRAM into a CB for tilization
 *
 * Reads total_num_rows sticks, grouping them into tile-height blocks.
 * Handles non-tile-aligned widths by padding the L1 stride.
 * Handles non-tile-aligned heights by pushing full tile pages for the
 * last partial block (untouched rows contain stale data).
 *
 * With TILE granularity: pushes width_in_tiles pages per block.
 *   CB must be configured with page_size = tile_size.
 *   Compute tilize: tilize(num_blocks)
 *
 * With ROW granularity: pushes 1 page per row.
 *   CB must be configured with page_size = padded_row_bytes.
 *   Compute tilize: tilize(num_blocks, total_num_rows)
 *
 * @tparam cb_id Circular buffer to write into (must be constexpr)
 * @tparam granularity TILE (default) or ROW
 * @tparam Accessor TensorAccessor type (deduced)
 * @param accessor TensorAccessor for the source tensor (stick-indexed)
 * @param total_num_rows Total number of sticks to read
 * @param row_bytes Actual bytes per stick (may be non-tile-aligned)
 */
template <uint32_t cb_id, TilizeGranularity granularity = TilizeGranularity::TILE, typename Accessor>
FORCE_INLINE void read_sticks_for_tilize(const Accessor& accessor, uint32_t total_num_rows, uint32_t row_bytes);

/**
 * @brief Write untilized sticks from a CB to DRAM
 *
 * Reads total_num_rows worth of untilized data from the CB (produced by
 * the untilize compute helper) and writes the valid sticks to DRAM.
 *
 * Handles non-tile-aligned widths by skipping L1 padding between rows.
 * Handles non-tile-aligned heights by popping full tile pages for the
 * last partial block but only writing the valid rows.
 *
 * Always operates at TILE granularity (untilize compute always produces
 * tile-sized pages).
 *
 * @tparam cb_id Circular buffer to read from (must be constexpr)
 * @tparam Accessor TensorAccessor type (deduced)
 * @param accessor TensorAccessor for the destination tensor (stick-indexed)
 * @param total_num_rows Total number of sticks to write
 * @param row_bytes Actual bytes per stick to write (may be non-tile-aligned)
 */
template <uint32_t cb_id, typename Accessor>
FORCE_INLINE void write_sticks_after_untilize(const Accessor& accessor, uint32_t total_num_rows, uint32_t row_bytes);

}  // namespace dataflow_kernel_lib

#include "ttnn/cpp/ttnn/kernel_lib/tilize_helpers_dataflow.inl"

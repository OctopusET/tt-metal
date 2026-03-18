// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "api/dataflow/dataflow_api.h"

namespace dataflow_kernel_lib {

/**
 * @brief Read row-major sticks from DRAM into a CB for tilization
 *
 * Reads total_num_rows sticks, grouping them into tile-height blocks.
 * Each block is pushed as width_in_tiles pages to the CB, ready for
 * the tilize compute helper.
 *
 * Handles non-tile-aligned widths by padding the L1 stride.
 * Handles non-tile-aligned heights by pushing full tile pages for the
 * last partial block (untouched rows contain stale data).
 *
 * @tparam cb_id Circular buffer to write into (must be constexpr)
 * @tparam Accessor TensorAccessor type (deduced)
 * @param accessor TensorAccessor for the source tensor (stick-indexed)
 * @param total_num_rows Total number of sticks to read
 * @param row_bytes Actual bytes per stick (may be non-tile-aligned)
 */
template <uint32_t cb_id, typename Accessor>
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

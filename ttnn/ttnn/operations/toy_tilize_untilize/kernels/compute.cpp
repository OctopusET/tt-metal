// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "ttnn/cpp/ttnn/kernel_lib/tilize_helpers.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/untilize_helpers.hpp"

void kernel_main() {
    constexpr uint32_t cb_rm_in = 0;
    constexpr uint32_t cb_tilized = 24;
    constexpr uint32_t cb_rm_out = 16;
    constexpr uint32_t width_tiles = get_compile_time_arg_val(0);
    constexpr uint32_t num_blocks = get_compile_time_arg_val(1);

    compute_kernel_hw_startup(cb_rm_in, cb_tilized);

    // Interleave tilize and untilize block-by-block so cb_tilized only needs
    // to hold one block at a time (double-buffered with the reader/writer).
    for (uint32_t block = 0; block < num_blocks; ++block) {
        compute_kernel_lib::tilize<
            width_tiles,
            cb_rm_in,
            cb_tilized,
            compute_kernel_lib::tilize_config::InitUninitMode::InitAndUninit,
            compute_kernel_lib::tilize_config::WaitMode::WaitBlock,
            compute_kernel_lib::tilize_config::ReconfigureRegisterDatatypeMode::NoReconfigure>(1);
        compute_kernel_lib::untilize<
            width_tiles,
            cb_tilized,
            cb_rm_out,
            compute_kernel_lib::untilize_config::InitUninitMode::InitAndUninit,
            compute_kernel_lib::untilize_config::WaitMode::WaitBlock,
            compute_kernel_lib::untilize_config::ReconfigureRegisterDatatypeMode::NoReconfigure>(1);
    }
}

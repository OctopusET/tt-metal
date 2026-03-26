// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "gated_delta_net_device_operation_types.hpp"
#include "ttnn/device_operation.hpp"

namespace ttnn::experimental::prim {

struct GatedDeltaNetSharedVariables {
    tt::tt_metal::KernelHandle compute_kernel_id = 0;
    tt::tt_metal::KernelHandle reader_kernel_id = 0;
    tt::tt_metal::KernelHandle writer_kernel_id = 0;
    std::vector<tt::tt_metal::CoreCoord> cores;
    uint32_t num_heads = 0;
    uint32_t head_dim = 0;
    uint32_t batch_size = 0;
};

struct GatedDeltaNetProgramFactory {
    using shared_variables_t = GatedDeltaNetSharedVariables;
    using cached_program_t = ttnn::device_operation::CachedProgram<shared_variables_t>;

    static cached_program_t create(
        const GatedDeltaNetParams& operation_attributes,
        const GatedDeltaNetInputs& tensor_args,
        std::vector<Tensor>& tensor_return_value);

    static void override_runtime_arguments(
        cached_program_t& cached_program,
        const GatedDeltaNetParams& operation_attributes,
        const GatedDeltaNetInputs& tensor_args,
        std::vector<Tensor>& tensor_return_value);
};

}  // namespace ttnn::experimental::prim

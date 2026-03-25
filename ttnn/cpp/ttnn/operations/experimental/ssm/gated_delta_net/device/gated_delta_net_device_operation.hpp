// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <functional>
#include <optional>

#include "ttnn/tensor/tensor.hpp"
#include "gated_delta_net_program_factory.hpp"
#include "ttnn/device_operation.hpp"
#include "ttnn/decorators.hpp"
#include "gated_delta_net_device_operation_types.hpp"

namespace ttnn::experimental::prim {

struct GatedDeltaNetDeviceOperation {
    using operation_attributes_t = GatedDeltaNetParams;
    using tensor_args_t = GatedDeltaNetInputs;
    using spec_return_value_t = std::vector<TensorSpec>;
    using tensor_return_value_t = std::vector<Tensor>;
    using program_factory_t = std::variant<GatedDeltaNetProgramFactory>;

    static void validate_on_program_cache_miss(const operation_attributes_t&, const tensor_args_t&);
    static spec_return_value_t compute_output_specs(const operation_attributes_t&, const tensor_args_t&);
    static tensor_return_value_t create_output_tensors(const operation_attributes_t&, const tensor_args_t&);
    static ttsl::hash::hash_t compute_program_hash(const operation_attributes_t&, const tensor_args_t&);
};

}  // namespace ttnn::experimental::prim

namespace ttnn::prim {

std::vector<Tensor> gated_delta_net(
    const Tensor& conv_out,
    const Tensor& z_flat,
    const Tensor& ba_flat,
    const Tensor& dt_bias,
    const Tensor& neg_A_exp,
    const Tensor& state,
    const Tensor& norm_weight,
    float scale = 1.0f,
    float norm_eps = 1e-6f,
    uint32_t key_dim = 2048,
    uint32_t gqa_ratio = 1,
    const std::optional<MemoryConfig>& memory_config = std::nullopt);

}  // namespace ttnn::prim

// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "gated_delta_net_device_operation.hpp"

#include "ttnn/tensor/tensor.hpp"
#include <tt-metalium/constants.hpp>

using namespace tt::tt_metal;

namespace ttnn::experimental::prim {

void GatedDeltaNetDeviceOperation::validate_on_program_cache_miss(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    const auto& conv_out = tensor_args.conv_out;
    const auto& state = tensor_args.state;

    TT_FATAL(conv_out.storage_type() == StorageType::DEVICE, "Inputs must be on device");
    TT_FATAL(conv_out.layout() == Layout::TILE, "Expected TILE layout");
    TT_FATAL(state.layout() == Layout::TILE, "Expected TILE layout for state");
    TT_FATAL(state.dtype() == DataType::FLOAT32 || state.dtype() == DataType::BFLOAT16, "State must be fp32 or bf16");
    TT_FATAL(args.key_dim > 0, "key_dim must be > 0");
    TT_FATAL(args.gqa_ratio > 0, "gqa_ratio must be > 0");
}

GatedDeltaNetDeviceOperation::spec_return_value_t GatedDeltaNetDeviceOperation::compute_output_specs(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    const auto& state = tensor_args.state;
    const auto& memory_config = args.memory_config;

    uint32_t batch_size = state.logical_shape()[0];
    uint32_t num_heads = state.logical_shape()[1];
    uint32_t head_dim = state.logical_shape()[3];
    auto out_shape = ttnn::Shape({1, num_heads, batch_size, head_dim});

    std::vector<TensorSpec> output_specs;
    output_specs.reserve(2);
    output_specs.push_back(
        TensorSpec(out_shape, TensorLayout(DataType::BFLOAT16, PageConfig(Layout::TILE), memory_config)));
    output_specs.push_back(
        TensorSpec(state.logical_shape(), TensorLayout(state.dtype(), PageConfig(Layout::TILE), memory_config)));

    return output_specs;
}

GatedDeltaNetDeviceOperation::tensor_return_value_t GatedDeltaNetDeviceOperation::create_output_tensors(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    const auto output_specs = compute_output_specs(args, tensor_args);
    auto* device = tensor_args.conv_out.device();

    std::vector<Tensor> output_tensors;
    output_tensors.reserve(output_specs.size());
    for (const auto& spec : output_specs) {
        output_tensors.push_back(create_device_tensor(spec, device));
    }
    return output_tensors;
}

ttsl::hash::hash_t GatedDeltaNetDeviceOperation::compute_program_hash(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    const auto& state = tensor_args.state;
    return operation::hash_operation<GatedDeltaNetDeviceOperation>(
        state.dtype(), state.memory_config(), state.padded_shape().volume(), args.key_dim, args.gqa_ratio);
}

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
    float scale,
    float norm_eps,
    uint32_t key_dim,
    uint32_t gqa_ratio,
    const std::optional<MemoryConfig>& memory_config) {
    using OpType = ttnn::experimental::prim::GatedDeltaNetDeviceOperation;
    auto mem_cfg = memory_config.value_or(conv_out.memory_config());
    auto attrs = OpType::operation_attributes_t{mem_cfg, scale, norm_eps, key_dim, gqa_ratio};
    auto inputs = OpType::tensor_args_t{conv_out, z_flat, ba_flat, dt_bias, neg_A_exp, state, norm_weight};
    return ttnn::device_operation::launch<OpType>(attrs, inputs);
}

}  // namespace ttnn::prim

// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "gated_delta_net_device_operation.hpp"

#include "ttnn/tensor/tensor.hpp"
#include <tt-metalium/constants.hpp>

using namespace tt::tt_metal;

namespace ttnn::experimental::prim {

void GatedDeltaNetDeviceOperation::validate_on_program_cache_miss(
    const operation_attributes_t& /*args*/, const tensor_args_t& tensor_args) {
    const auto& q = tensor_args.q;
    const auto& state = tensor_args.state;

    TT_FATAL(q.storage_type() == StorageType::DEVICE, "Inputs must be on device");
    TT_FATAL(q.layout() == Layout::TILE, "Expected TILE layout");
    TT_FATAL(state.layout() == Layout::TILE, "Expected TILE layout for state");
    TT_FATAL(state.dtype() == DataType::FLOAT32, "State must be fp32");
    TT_FATAL(q.memory_config().memory_layout() == TensorMemoryLayout::INTERLEAVED, "Expected interleaved tensors");
    TT_FATAL(q.padded_shape()[1] == state.padded_shape()[1], "Q and state must have same num_heads");
}

GatedDeltaNetDeviceOperation::spec_return_value_t GatedDeltaNetDeviceOperation::compute_output_specs(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    const auto& q = tensor_args.q;
    const auto& state = tensor_args.state;
    const auto& memory_config = args.memory_config;

    std::vector<TensorSpec> output_specs;
    output_specs.reserve(2);

    // output: same shape as q, bf16
    output_specs.push_back(
        TensorSpec(q.logical_shape(), TensorLayout(DataType::BFLOAT16, PageConfig(Layout::TILE), memory_config)));

    // new_state: same shape as state, fp32
    output_specs.push_back(
        TensorSpec(state.logical_shape(), TensorLayout(DataType::FLOAT32, PageConfig(Layout::TILE), memory_config)));

    return output_specs;
}

GatedDeltaNetDeviceOperation::tensor_return_value_t GatedDeltaNetDeviceOperation::create_output_tensors(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    const auto output_specs = compute_output_specs(args, tensor_args);
    auto* device = tensor_args.q.device();

    std::vector<Tensor> output_tensors;
    output_tensors.reserve(output_specs.size());
    for (const auto& spec : output_specs) {
        output_tensors.push_back(create_device_tensor(spec, device));
    }
    return output_tensors;
}

ttsl::hash::hash_t GatedDeltaNetDeviceOperation::compute_program_hash(
    const operation_attributes_t& /*args*/, const tensor_args_t& tensor_args) {
    const auto& q = tensor_args.q;
    const auto& state = tensor_args.state;
    return operation::hash_operation<GatedDeltaNetDeviceOperation>(
        q.dtype(), state.dtype(), q.memory_config(), q.padded_shape().volume(), state.padded_shape().volume());
}

}  // namespace ttnn::experimental::prim

namespace ttnn::prim {

std::vector<Tensor> gated_delta_net(
    const Tensor& q,
    const Tensor& k,
    const Tensor& v,
    const Tensor& decay,
    const Tensor& beta,
    const Tensor& state,
    float scale,
    const std::optional<MemoryConfig>& memory_config) {
    using OpType = ttnn::experimental::prim::GatedDeltaNetDeviceOperation;
    auto mem_cfg = memory_config.value_or(q.memory_config());
    auto attrs = OpType::operation_attributes_t{mem_cfg, scale};
    auto inputs = OpType::tensor_args_t{q, k, v, decay, beta, state};
    return ttnn::device_operation::launch<OpType>(attrs, inputs);
}

}  // namespace ttnn::prim

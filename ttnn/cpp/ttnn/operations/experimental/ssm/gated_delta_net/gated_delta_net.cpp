// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "gated_delta_net.hpp"
#include "device/gated_delta_net_device_operation.hpp"

namespace ttnn::experimental {

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
    return ttnn::prim::gated_delta_net(
        conv_out,
        z_flat,
        ba_flat,
        dt_bias,
        neg_A_exp,
        state,
        norm_weight,
        scale,
        norm_eps,
        key_dim,
        gqa_ratio,
        memory_config);
}

}  // namespace ttnn::experimental

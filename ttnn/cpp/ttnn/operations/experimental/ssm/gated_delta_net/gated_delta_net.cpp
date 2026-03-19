// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "gated_delta_net.hpp"
#include "device/gated_delta_net_device_operation.hpp"

namespace ttnn::experimental {

std::vector<Tensor> gated_delta_net(
    const Tensor& q,
    const Tensor& k,
    const Tensor& v,
    const Tensor& decay,
    const Tensor& beta,
    const Tensor& state,
    float scale,
    const std::optional<MemoryConfig>& memory_config) {
    return ttnn::prim::gated_delta_net(q, k, v, decay, beta, state, scale, memory_config);
}

}  // namespace ttnn::experimental

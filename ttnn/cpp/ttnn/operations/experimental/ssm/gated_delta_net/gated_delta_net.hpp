// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/tensor/tensor.hpp"

namespace ttnn::experimental {

// Fused GatedDeltaNet recurrence: decay + retrieve + delta + write + read.
// Returns [output (1,H,1,D) bf16, new_state (1,H,D,D) fp32].
std::vector<Tensor> gated_delta_net(
    const Tensor& q,
    const Tensor& k,
    const Tensor& v,
    const Tensor& decay,
    const Tensor& beta,
    const Tensor& state,
    float scale = 1.0f,
    const std::optional<tt::tt_metal::MemoryConfig>& memory_config = std::nullopt);

}  // namespace ttnn::experimental

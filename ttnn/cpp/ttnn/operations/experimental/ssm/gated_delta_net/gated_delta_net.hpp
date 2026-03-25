// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/tensor/tensor.hpp"

namespace ttnn::experimental {

// Fused GatedDeltaNet: L2 norm + gates + retrieve + delta + recurrence + gated RMSNorm.
// conv_out = flat conv output (1,1,B,conv_dim), z_flat = (1,1,B,value_dim), ba_flat = (1,1,B,2H).
// Reader reads per-head tiles using key_dim/gqa_ratio offsets, constructs k_col.
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
    const std::optional<tt::tt_metal::MemoryConfig>& memory_config = std::nullopt);

}  // namespace ttnn::experimental

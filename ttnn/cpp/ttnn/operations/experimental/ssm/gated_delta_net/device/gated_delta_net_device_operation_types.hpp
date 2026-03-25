// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/tensor/tensor.hpp"
#include <tt-metalium/base_types.hpp>

namespace ttnn::experimental::prim {

struct GatedDeltaNetParams {
    const tt::tt_metal::MemoryConfig memory_config;
    const float scale;
    const float norm_eps;
    const uint32_t key_dim;    // key_dim = num_k_heads * head_dim
    const uint32_t gqa_ratio;  // num_v_heads / num_k_heads
};

struct GatedDeltaNetInputs {
    Tensor conv_out;     // (1, 1, B, conv_dim) bf16 - flat conv output (q|k|v concatenated)
    Tensor z_flat;       // (1, 1, B, value_dim) bf16 - flat z gate input
    Tensor ba_flat;      // (1, 1, B, 2*H) bf16 - packed b|a (2 tiles)
    Tensor dt_bias;      // (1, H, 1, 1) bf16 - time-delta bias (constant per layer)
    Tensor neg_A_exp;    // (1, H, 1, 1) bf16 - negative decay rate: -exp(A_log)
    Tensor state;        // (1, H, D, D) fp32 - recurrent state
    Tensor norm_weight;  // (1, H, 1, D) bf16 - RMSNorm learnable weight
};

}  // namespace ttnn::experimental::prim

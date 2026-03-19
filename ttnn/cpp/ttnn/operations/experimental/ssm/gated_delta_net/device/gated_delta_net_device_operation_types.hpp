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
};

struct GatedDeltaNetInputs {
    Tensor q;      // (1, H, 1, D) bf16 - L2-normalized, scaled query
    Tensor k;      // (1, H, 1, D) bf16 - L2-normalized key
    Tensor v;      // (1, H, 1, D) bf16 - value
    Tensor decay;  // (1, H, 1, 1) bf16 - exp(gate), decay factor per head
    Tensor beta;   // (1, H, 1, 1) bf16 - sigmoid(b), update strength per head
    Tensor state;  // (1, H, D, D) fp32 - recurrent state
};

}  // namespace ttnn::experimental::prim

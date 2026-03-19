// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "gated_delta_net_nanobind.hpp"
#include "gated_delta_net.hpp"

#include "ttnn/cpp/ttnn/decorators.hpp"

namespace nb = nanobind;

namespace ttnn::operations::experimental::ssm::detail {

void bind_gated_delta_net(nb::module_& mod) {
    const auto* const doc = R"doc(
        Fused GatedDeltaNet recurrence operation.

        Performs the full DeltaNet recurrence (decay, retrieve, delta, write, read)
        in a single fused kernel with fp32 state accumulation.

        Args:
            q: Query tensor (1, H, 1, D) bf16
            k: Key tensor (1, H, 1, D) bf16
            v: Value tensor (1, H, 1, D) bf16
            decay: Decay factor (1, H, 1, 1) bf16
            beta: Update strength (1, H, 1, 1) bf16
            state: Recurrent state (1, H, D, D) fp32
            scale: Query scale factor (default 1.0)

        Returns:
            [output (1, H, 1, D) bf16, new_state (1, H, D, D) fp32]
    )doc";

    ttnn::bind_function<"gated_delta_net", "ttnn.experimental.">(
        mod,
        doc,
        &ttnn::experimental::gated_delta_net,
        nb::arg("q"),
        nb::arg("k"),
        nb::arg("v"),
        nb::arg("decay"),
        nb::arg("beta"),
        nb::arg("state"),
        nb::kw_only(),
        nb::arg("scale") = 1.0f,
        nb::arg("memory_config") = nb::none());
}

}  // namespace ttnn::operations::experimental::ssm::detail

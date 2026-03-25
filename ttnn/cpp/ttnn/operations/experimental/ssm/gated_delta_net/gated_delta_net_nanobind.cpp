// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "gated_delta_net_nanobind.hpp"

#include <optional>

#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/vector.h>

#include "ttnn-nanobind/bind_function.hpp"
#include "gated_delta_net.hpp"

namespace ttnn::operations::experimental::ssm::detail {

void bind_gated_delta_net(nb::module_& mod) {
    const auto* const doc =
        R"doc(Fused GatedDeltaNet recurrence: decay + retrieve + delta + write + read in one kernel with fp32 state.)doc";

    ttnn::bind_function<"gated_delta_net", "ttnn.experimental.">(
        mod,
        doc,
        &ttnn::experimental::gated_delta_net,
        nb::arg("conv_out"),
        nb::arg("z_flat"),
        nb::arg("ba_flat"),
        nb::arg("dt_bias"),
        nb::arg("neg_A_exp"),
        nb::arg("state"),
        nb::arg("norm_weight"),
        nb::kw_only(),
        nb::arg("scale") = 1.0f,
        nb::arg("norm_eps") = 1e-6f,
        nb::arg("key_dim") = 2048,
        nb::arg("gqa_ratio") = 1,
        nb::arg("memory_config") = nb::none());
}

}  // namespace ttnn::operations::experimental::ssm::detail

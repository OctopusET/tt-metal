// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "gated_delta_net_nanobind.hpp"

#include <optional>

#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>

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

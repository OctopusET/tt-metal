// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "gated_delta_net_program_factory.hpp"

#include "ttnn/tensor/tensor.hpp"
#include <tt-metalium/constants.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/work_split.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>

using namespace tt::tt_metal;

namespace ttnn::experimental::prim {

using namespace tt::constants;

GatedDeltaNetProgramFactory::cached_program_t GatedDeltaNetProgramFactory::create(
    const GatedDeltaNetParams& operation_attributes,
    const GatedDeltaNetInputs& tensor_args,
    std::vector<Tensor>& tensor_return_value) {
    [[maybe_unused]] auto& output = tensor_return_value[0];
    [[maybe_unused]] auto& new_state = tensor_return_value[1];
    tt::tt_metal::Program program = tt::tt_metal::CreateProgram();

    const auto& q = tensor_args.q;

    // Shapes: q,k,v = (1, H, 1, D), decay,beta = (1, H, 1, 1), state = (1, H, D, D)
    const uint32_t num_heads = q.padded_shape()[1];
    const uint32_t head_dim = q.padded_shape()[3];   // D = 128
    const uint32_t D_TILES = head_dim / TILE_WIDTH;  // 4
    const uint32_t STATE_TILES = D_TILES * D_TILES;  // 16

    // Data formats
    const tt::DataFormat bf16_format = tt::DataFormat::Float16_b;
    const uint32_t bf16_tile_size = tt::tile_size(bf16_format);

    // Core mapping: 1 head per core
    auto device = q.device();
    auto grid_size = device->compute_with_storage_grid_size();
    auto [num_cores, all_cores, core_group_1, core_group_2, heads_per_core_g1, heads_per_core_g2] =
        tt::tt_metal::split_work_to_cores(grid_size, num_heads, /*row_major=*/true);

    // CB creation helper
    const auto create_cb = [&program, &all_cores](
                               uint32_t index,
                               uint32_t num_tiles,
                               uint32_t tile_size,
                               const tt::DataFormat& format) -> tt::tt_metal::CBHandle {
        auto config = CircularBufferConfig(num_tiles * tile_size, {{index, format}}).set_page_size(index, tile_size);
        return tt::tt_metal::CreateCircularBuffer(program, all_cores, config);
    };

    // Circular buffers -- all bf16 (fp32 CBs hang with element-wise ops on Blackhole)
    // fp32_dest_acc_en provides fp32 accumulation in dest registers
    const uint32_t cb_q_id = tt::CBIndex::c_0;          // Q vector [D_TILES]
    const uint32_t cb_k_id = tt::CBIndex::c_1;          // K vector [D_TILES]
    const uint32_t cb_v_id = tt::CBIndex::c_2;          // V vector [D_TILES]
    const uint32_t cb_decay_id = tt::CBIndex::c_3;      // decay scalar [1]
    const uint32_t cb_beta_id = tt::CBIndex::c_4;       // beta scalar [1]
    const uint32_t cb_state_id = tt::CBIndex::c_5;      // input state [STATE_TILES] (reader->compute)
    const uint32_t cb_state_new_id = tt::CBIndex::c_6;  // output state [STATE_TILES] (compute->writer)
    const uint32_t cb_out_id = tt::CBIndex::c_7;        // output vector [D_TILES] (compute->writer)
    const uint32_t cb_sd_id = tt::CBIndex::c_24;        // bf16 state for fp32->bf16 conversion
    const uint32_t cb_tmp_id = tt::CBIndex::c_25;       // scratch [D_TILES]
    const uint32_t cb_tmp2_id = tt::CBIndex::c_26;      // scratch2 [D_TILES]
    const uint32_t cb_sd2_id = tt::CBIndex::c_27;       // bf16 decayed state for add in Phase 3

    // State CBs: fp32 for precision. matmul_tiles works with fp32 CBs.
    // Element-wise ops use bf16 intermediates (cb_sd, cb_sd2).
    const tt::DataFormat fp32_format = tt::DataFormat::Float32;
    const uint32_t fp32_tile_size = tt::tile_size(fp32_format);

    create_cb(cb_q_id, D_TILES, bf16_tile_size, bf16_format);
    create_cb(cb_k_id, D_TILES, bf16_tile_size, bf16_format);
    create_cb(cb_v_id, D_TILES, bf16_tile_size, bf16_format);
    create_cb(cb_decay_id, 1, bf16_tile_size, bf16_format);
    create_cb(cb_beta_id, 1, bf16_tile_size, bf16_format);
    create_cb(cb_state_id, STATE_TILES, fp32_tile_size, fp32_format);      // fp32 input state
    create_cb(cb_state_new_id, STATE_TILES, fp32_tile_size, fp32_format);  // fp32 output state
    create_cb(cb_out_id, D_TILES, bf16_tile_size, bf16_format);
    create_cb(cb_sd_id, STATE_TILES, bf16_tile_size, bf16_format);  // bf16 conversion temp
    create_cb(cb_tmp_id, D_TILES, bf16_tile_size, bf16_format);
    create_cb(cb_tmp2_id, D_TILES, bf16_tile_size, bf16_format);
    create_cb(cb_sd2_id, STATE_TILES, bf16_tile_size, bf16_format);  // bf16 decayed for add

    // Get buffers for TensorAccessorArgs
    auto* q_buffer = tensor_args.q.buffer();
    auto* k_buffer = tensor_args.k.buffer();
    auto* v_buffer = tensor_args.v.buffer();
    auto* decay_buffer = tensor_args.decay.buffer();
    auto* beta_buffer = tensor_args.beta.buffer();
    auto* state_buffer = tensor_args.state.buffer();
    auto* output_buffer = output.buffer();
    auto* new_state_buffer = new_state.buffer();

    // Compile-time args for reader (CB ids + dims + TensorAccessorArgs for 6 inputs)
    std::vector<uint32_t> reader_ct_args = {
        cb_q_id, cb_k_id, cb_v_id, cb_decay_id, cb_beta_id, cb_state_id, D_TILES, STATE_TILES};
    tt::tt_metal::TensorAccessorArgs(q_buffer).append_to(reader_ct_args);
    tt::tt_metal::TensorAccessorArgs(k_buffer).append_to(reader_ct_args);
    tt::tt_metal::TensorAccessorArgs(v_buffer).append_to(reader_ct_args);
    tt::tt_metal::TensorAccessorArgs(decay_buffer).append_to(reader_ct_args);
    tt::tt_metal::TensorAccessorArgs(beta_buffer).append_to(reader_ct_args);
    tt::tt_metal::TensorAccessorArgs(state_buffer).append_to(reader_ct_args);

    // Compile-time args for writer (CB ids + dims + TensorAccessorArgs for 2 outputs)
    std::vector<uint32_t> writer_ct_args = {cb_out_id, cb_state_new_id, D_TILES, STATE_TILES};
    tt::tt_metal::TensorAccessorArgs(output_buffer).append_to(writer_ct_args);
    tt::tt_metal::TensorAccessorArgs(new_state_buffer).append_to(writer_ct_args);
    std::vector<uint32_t> compute_ct_args = {
        cb_q_id,
        cb_k_id,
        cb_v_id,
        cb_decay_id,
        cb_beta_id,
        cb_state_id,
        cb_state_new_id,
        cb_out_id,
        cb_sd_id,
        cb_tmp_id,
        cb_tmp2_id,
        cb_sd2_id,
        D_TILES,
        STATE_TILES};

    // Create kernels
    auto reader_kernel_id = tt::tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/experimental/ssm/gated_delta_net/device/kernels/reader_gated_delta_net.cpp",
        all_cores,
        tt::tt_metal::ReaderDataMovementConfig(reader_ct_args));

    auto writer_kernel_id = tt::tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/experimental/ssm/gated_delta_net/device/kernels/writer_gated_delta_net.cpp",
        all_cores,
        tt::tt_metal::WriterDataMovementConfig(writer_ct_args));

    auto compute_kernel_id = tt::tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/experimental/ssm/gated_delta_net/device/kernels/gated_delta_net_compute.cpp",
        all_cores,
        tt::tt_metal::ComputeConfig{
            .math_fidelity = MathFidelity::HiFi4,
            .fp32_dest_acc_en = true,
            .math_approx_mode = false,
            .compile_args = compute_ct_args});

    // Store shared variables
    GatedDeltaNetSharedVariables shared_variables;
    shared_variables.compute_kernel_id = compute_kernel_id;
    shared_variables.reader_kernel_id = reader_kernel_id;
    shared_variables.writer_kernel_id = writer_kernel_id;
    shared_variables.num_heads = num_heads;
    shared_variables.head_dim = head_dim;
    shared_variables.cores = grid_to_cores(num_cores, grid_size.x, grid_size.y, true);

    cached_program_t cached_program{std::move(program), std::move(shared_variables)};

    // Set runtime args
    override_runtime_arguments(cached_program, operation_attributes, tensor_args, tensor_return_value);

    return cached_program;
}

void GatedDeltaNetProgramFactory::override_runtime_arguments(
    cached_program_t& cached_program,
    const GatedDeltaNetParams& /*operation_attributes*/,
    const GatedDeltaNetInputs& tensor_args,
    std::vector<Tensor>& tensor_return_value) {
    auto& output = tensor_return_value[0];
    auto& new_state = tensor_return_value[1];
    auto& program = cached_program.program;
    const auto& sv = cached_program.shared_variables;

    // Set per-core runtime args: buffer addresses + head index
    for (uint32_t i = 0; i < sv.cores.size(); i++) {
        uint32_t head_start = i;
        uint32_t num_heads_this_core = 1;

        SetRuntimeArgs(
            program,
            sv.reader_kernel_id,
            sv.cores[i],
            {
                tensor_args.q.buffer()->address(),
                tensor_args.k.buffer()->address(),
                tensor_args.v.buffer()->address(),
                tensor_args.decay.buffer()->address(),
                tensor_args.beta.buffer()->address(),
                tensor_args.state.buffer()->address(),
                head_start,
                num_heads_this_core,
            });

        SetRuntimeArgs(
            program,
            sv.writer_kernel_id,
            sv.cores[i],
            {
                output.buffer()->address(),
                new_state.buffer()->address(),
                head_start,
                num_heads_this_core,
            });

        SetRuntimeArgs(
            program,
            sv.compute_kernel_id,
            sv.cores[i],
            {
                num_heads_this_core,
            });
    }
}

}  // namespace ttnn::experimental::prim

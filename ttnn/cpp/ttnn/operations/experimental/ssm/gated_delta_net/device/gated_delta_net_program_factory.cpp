// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "gated_delta_net_program_factory.hpp"

#include "ttnn/tensor/tensor.hpp"
#include <tt-metalium/constants.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/work_split.hpp>

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
    const tt::DataFormat fp32_format = tt::DataFormat::Float32;
    const uint32_t bf16_tile_size = tt::tile_size(bf16_format);
    const uint32_t fp32_tile_size = tt::tile_size(fp32_format);

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

    // Circular buffers (192 KB total per core)
    const uint32_t cb_q_id = tt::CBIndex::c_0;
    const uint32_t cb_k_id = tt::CBIndex::c_1;
    const uint32_t cb_v_id = tt::CBIndex::c_2;
    const uint32_t cb_decay_id = tt::CBIndex::c_3;
    const uint32_t cb_beta_id = tt::CBIndex::c_4;
    const uint32_t cb_state_id = tt::CBIndex::c_5;
    const uint32_t cb_state_new_id = tt::CBIndex::c_6;
    const uint32_t cb_out_id = tt::CBIndex::c_7;
    const uint32_t cb_kv_mem_id = tt::CBIndex::c_24;
    const uint32_t cb_delta_id = tt::CBIndex::c_25;
    const uint32_t cb_k_t_id = tt::CBIndex::c_26;

    create_cb(cb_q_id, D_TILES, bf16_tile_size, bf16_format);
    create_cb(cb_k_id, D_TILES, bf16_tile_size, bf16_format);
    create_cb(cb_v_id, D_TILES, bf16_tile_size, bf16_format);
    create_cb(cb_decay_id, 1, bf16_tile_size, bf16_format);
    create_cb(cb_beta_id, 1, bf16_tile_size, bf16_format);
    create_cb(cb_state_id, STATE_TILES, fp32_tile_size, fp32_format);
    create_cb(cb_state_new_id, STATE_TILES, fp32_tile_size, fp32_format);
    create_cb(cb_out_id, D_TILES, bf16_tile_size, bf16_format);
    create_cb(cb_kv_mem_id, D_TILES, bf16_tile_size, bf16_format);
    create_cb(cb_delta_id, D_TILES, bf16_tile_size, bf16_format);
    create_cb(cb_k_t_id, D_TILES, bf16_tile_size, bf16_format);

    // Compile-time args for kernels
    std::vector<uint32_t> reader_ct_args = {
        cb_q_id, cb_k_id, cb_v_id, cb_decay_id, cb_beta_id, cb_state_id, D_TILES, STATE_TILES};
    std::vector<uint32_t> writer_ct_args = {cb_out_id, cb_state_new_id, D_TILES, STATE_TILES};
    std::vector<uint32_t> compute_ct_args = {
        cb_q_id,
        cb_k_id,
        cb_v_id,
        cb_decay_id,
        cb_beta_id,
        cb_state_id,
        cb_state_new_id,
        cb_out_id,
        cb_kv_mem_id,
        cb_delta_id,
        cb_k_t_id,
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
    [[maybe_unused]] auto& output = tensor_return_value[0];
    [[maybe_unused]] auto& new_state = tensor_return_value[1];
    auto& program = cached_program.program;
    const auto& sv = cached_program.shared_variables;

    auto* q_buffer = tensor_args.q.buffer();
    auto* k_buffer = tensor_args.k.buffer();
    auto* v_buffer = tensor_args.v.buffer();
    auto* decay_buffer = tensor_args.decay.buffer();
    auto* beta_buffer = tensor_args.beta.buffer();
    auto* state_buffer = tensor_args.state.buffer();
    auto* output_buffer = output.buffer();
    auto* new_state_buffer = new_state.buffer();

    [[maybe_unused]] const uint32_t D_TILES = sv.head_dim / TILE_WIDTH;

    // bf16 and fp32 tile sizes
    const uint32_t bf16_tile_size = tt::tile_size(tt::DataFormat::Float16_b);
    const uint32_t fp32_tile_size = tt::tile_size(tt::DataFormat::Float32);

    // Set per-core runtime args: buffer base addresses + head offset
    for (uint32_t i = 0; i < sv.cores.size(); i++) {
        // Reader: base addresses + head_start + num_heads_this_core
        uint32_t head_start = i;  // 1 head per core (simplified)
        uint32_t num_heads_this_core = 1;

        SetRuntimeArgs(
            program,
            sv.reader_kernel_id,
            sv.cores[i],
            {
                q_buffer->address(),
                k_buffer->address(),
                v_buffer->address(),
                decay_buffer->address(),
                beta_buffer->address(),
                state_buffer->address(),
                head_start,
                num_heads_this_core,
                bf16_tile_size,
                fp32_tile_size,
            });

        SetRuntimeArgs(
            program,
            sv.writer_kernel_id,
            sv.cores[i],
            {
                output_buffer->address(),
                new_state_buffer->address(),
                head_start,
                num_heads_this_core,
                bf16_tile_size,
                fp32_tile_size,
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

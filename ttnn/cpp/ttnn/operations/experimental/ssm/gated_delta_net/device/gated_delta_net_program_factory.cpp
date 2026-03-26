// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "gated_delta_net_program_factory.hpp"

#include "ttnn/tensor/tensor.hpp"
#include <tt-metalium/constants.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/work_split.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include <cstring>

using namespace tt::tt_metal;

namespace ttnn::experimental::prim {

using namespace tt::constants;

static uint32_t pack_bf16_pair(float val) {
    uint32_t f32;
    std::memcpy(&f32, &val, sizeof(f32));
    uint16_t bf16 = static_cast<uint16_t>(f32 >> 16);
    return (static_cast<uint32_t>(bf16) << 16) | bf16;
}

GatedDeltaNetProgramFactory::cached_program_t GatedDeltaNetProgramFactory::create(
    const GatedDeltaNetParams& operation_attributes,
    const GatedDeltaNetInputs& tensor_args,
    std::vector<Tensor>& tensor_return_value) {
    [[maybe_unused]] auto& output = tensor_return_value[0];
    [[maybe_unused]] auto& new_state = tensor_return_value[1];
    tt::tt_metal::Program program = tt::tt_metal::CreateProgram();

    const auto& state = tensor_args.state;

    const uint32_t BATCH_SIZE = state.padded_shape()[0];
    const uint32_t num_heads = state.padded_shape()[1];
    const uint32_t head_dim = state.padded_shape()[3];
    const uint32_t D_TILES = head_dim / TILE_WIDTH;
    const uint32_t STATE_TILES = D_TILES * D_TILES;
    const uint32_t KEY_DIM_TILES = operation_attributes.key_dim / TILE_WIDTH;
    const uint32_t GQA_RATIO = operation_attributes.gqa_ratio;
    const uint32_t STATE_BATCH_STRIDE = num_heads * STATE_TILES;

    const tt::DataFormat bf16_format = tt::DataFormat::Float16_b;
    const uint32_t bf16_tile_size = tt::tile_size(bf16_format);
    const tt::DataFormat fp32_format = tt::DataFormat::Float32;
    const uint32_t fp32_tile_size = tt::tile_size(fp32_format);

    auto device = tensor_args.conv_out.device();
    auto grid_size = device->compute_with_storage_grid_size();
    auto [num_cores, all_cores, core_group_1, core_group_2, heads_per_core_g1, heads_per_core_g2] =
        tt::tt_metal::split_work_to_cores(grid_size, num_heads, /*row_major=*/true);

    const auto create_cb = [&program, &all_cores](
                               uint32_t index,
                               uint32_t num_tiles,
                               uint32_t tile_size,
                               const tt::DataFormat& format) -> tt::tt_metal::CBHandle {
        auto config = CircularBufferConfig(num_tiles * tile_size, {{index, format}}).set_page_size(index, tile_size);
        return tt::tt_metal::CreateCircularBuffer(program, all_cores, config);
    };

    // CB allocation (compute kernel unchanged)
    const uint32_t cb_q_id = tt::CBIndex::c_0;
    const uint32_t cb_k_col_id = tt::CBIndex::c_1;
    const uint32_t cb_v_id = tt::CBIndex::c_2;
    const uint32_t cb_decay_id = tt::CBIndex::c_3;
    const uint32_t cb_beta_id = tt::CBIndex::c_4;
    const uint32_t cb_state_id = tt::CBIndex::c_5;
    const uint32_t cb_state_new_id = tt::CBIndex::c_6;
    const uint32_t cb_out_id = tt::CBIndex::c_7;
    const uint32_t cb_k_row_id = tt::CBIndex::c_8;
    const uint32_t cb_b_id = tt::CBIndex::c_9;
    const uint32_t cb_a_id = tt::CBIndex::c_10;
    const uint32_t cb_dt_bias_id = tt::CBIndex::c_11;
    const uint32_t cb_neg_A_exp_id = tt::CBIndex::c_12;
    const uint32_t cb_kv_mem_id = tt::CBIndex::c_13;
    const uint32_t cb_delta_id = tt::CBIndex::c_14;
    const uint32_t cb_z_id = tt::CBIndex::c_15;
    const uint32_t cb_norm_w_id = tt::CBIndex::c_16;
    const uint32_t cb_scaler_id = tt::CBIndex::c_17;
    const uint32_t cb_eps_id = tt::CBIndex::c_18;
    const uint32_t cb_q_norm_id = tt::CBIndex::c_19;
    const uint32_t cb_k_row_norm_id = tt::CBIndex::c_20;
    const uint32_t cb_k_col_norm_id = tt::CBIndex::c_21;
    const uint32_t cb_scaler_one_id = tt::CBIndex::c_22;
    const uint32_t cb_q_scale_id = tt::CBIndex::c_23;
    const uint32_t cb_sd_id = tt::CBIndex::c_24;
    const uint32_t cb_tmp_id = tt::CBIndex::c_25;
    const uint32_t cb_tmp2_id = tt::CBIndex::c_26;
    const uint32_t cb_sd2_id = tt::CBIndex::c_27;
    const uint32_t cb_out_accum_id = tt::CBIndex::c_28;

    create_cb(cb_q_id, D_TILES, bf16_tile_size, bf16_format);
    create_cb(cb_k_col_id, D_TILES, bf16_tile_size, bf16_format);
    create_cb(cb_v_id, D_TILES, bf16_tile_size, bf16_format);
    create_cb(cb_decay_id, 1, bf16_tile_size, bf16_format);
    create_cb(cb_beta_id, 1, bf16_tile_size, bf16_format);
    create_cb(cb_state_id, STATE_TILES, fp32_tile_size, fp32_format);
    create_cb(cb_state_new_id, STATE_TILES, fp32_tile_size, fp32_format);
    create_cb(cb_out_id, D_TILES, bf16_tile_size, bf16_format);
    create_cb(cb_k_row_id, D_TILES, bf16_tile_size, bf16_format);
    create_cb(cb_b_id, 1, bf16_tile_size, bf16_format);
    create_cb(cb_a_id, 1, bf16_tile_size, bf16_format);
    create_cb(cb_dt_bias_id, 1, bf16_tile_size, bf16_format);
    create_cb(cb_neg_A_exp_id, 1, bf16_tile_size, bf16_format);
    create_cb(cb_kv_mem_id, D_TILES, bf16_tile_size, bf16_format);
    create_cb(cb_delta_id, D_TILES, bf16_tile_size, bf16_format);
    create_cb(cb_z_id, D_TILES, bf16_tile_size, bf16_format);
    create_cb(cb_norm_w_id, D_TILES, bf16_tile_size, bf16_format);
    create_cb(cb_scaler_id, 1, bf16_tile_size, bf16_format);
    create_cb(cb_eps_id, 1, bf16_tile_size, bf16_format);
    create_cb(cb_q_norm_id, D_TILES, bf16_tile_size, bf16_format);
    create_cb(cb_k_row_norm_id, D_TILES, bf16_tile_size, bf16_format);
    create_cb(cb_k_col_norm_id, D_TILES, bf16_tile_size, bf16_format);
    create_cb(cb_scaler_one_id, 1, bf16_tile_size, bf16_format);
    create_cb(cb_q_scale_id, 1, bf16_tile_size, bf16_format);
    create_cb(cb_sd_id, STATE_TILES, bf16_tile_size, bf16_format);
    create_cb(cb_tmp_id, D_TILES, bf16_tile_size, bf16_format);
    create_cb(cb_tmp2_id, D_TILES, bf16_tile_size, bf16_format);
    create_cb(cb_sd2_id, STATE_TILES, bf16_tile_size, bf16_format);
    create_cb(cb_out_accum_id, D_TILES, bf16_tile_size, bf16_format);

    auto* conv_out_buffer = tensor_args.conv_out.buffer();
    auto* z_flat_buffer = tensor_args.z_flat.buffer();
    auto* ba_flat_buffer = tensor_args.ba_flat.buffer();
    auto* dt_bias_buffer = tensor_args.dt_bias.buffer();
    auto* neg_A_exp_buffer = tensor_args.neg_A_exp.buffer();
    auto* state_buffer = tensor_args.state.buffer();
    auto* norm_w_buffer = tensor_args.norm_weight.buffer();
    auto* output_buffer = output.buffer();
    auto* new_state_buffer = new_state.buffer();

    // Reader compile-time args
    std::vector<uint32_t> reader_ct_args = {
        cb_q_id,         cb_k_col_id,      cb_v_id,       cb_k_row_id,   cb_b_id,   cb_a_id,      cb_dt_bias_id,
        cb_neg_A_exp_id, cb_state_id,      D_TILES,       STATE_TILES,   cb_z_id,   cb_norm_w_id, cb_scaler_id,
        cb_eps_id,       cb_scaler_one_id, cb_q_scale_id, KEY_DIM_TILES, GQA_RATIO, BATCH_SIZE,   STATE_BATCH_STRIDE};
    tt::tt_metal::TensorAccessorArgs(conv_out_buffer).append_to(reader_ct_args);
    tt::tt_metal::TensorAccessorArgs(z_flat_buffer).append_to(reader_ct_args);
    tt::tt_metal::TensorAccessorArgs(ba_flat_buffer).append_to(reader_ct_args);
    tt::tt_metal::TensorAccessorArgs(dt_bias_buffer).append_to(reader_ct_args);
    tt::tt_metal::TensorAccessorArgs(neg_A_exp_buffer).append_to(reader_ct_args);
    tt::tt_metal::TensorAccessorArgs(state_buffer).append_to(reader_ct_args);
    tt::tt_metal::TensorAccessorArgs(norm_w_buffer).append_to(reader_ct_args);

    // Writer compile-time args
    std::vector<uint32_t> writer_ct_args = {
        cb_out_id, cb_state_new_id, D_TILES, STATE_TILES, BATCH_SIZE, STATE_BATCH_STRIDE, cb_out_accum_id};
    tt::tt_metal::TensorAccessorArgs(output_buffer).append_to(writer_ct_args);
    tt::tt_metal::TensorAccessorArgs(new_state_buffer).append_to(writer_ct_args);

    // Compute compile-time args (unchanged)
    std::vector<uint32_t> compute_ct_args = {
        cb_q_id,         cb_k_col_id,  cb_v_id,          cb_decay_id,      cb_beta_id,       cb_state_id,
        cb_state_new_id, cb_out_id,    cb_sd_id,         cb_tmp_id,        cb_tmp2_id,       cb_sd2_id,
        D_TILES,         STATE_TILES,  cb_k_row_id,      cb_b_id,          cb_a_id,          cb_dt_bias_id,
        cb_neg_A_exp_id, cb_kv_mem_id, cb_delta_id,      cb_z_id,          cb_norm_w_id,     cb_scaler_id,
        cb_eps_id,       cb_q_norm_id, cb_k_row_norm_id, cb_k_col_norm_id, cb_scaler_one_id, cb_q_scale_id};

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

    GatedDeltaNetSharedVariables shared_variables;
    shared_variables.compute_kernel_id = compute_kernel_id;
    shared_variables.reader_kernel_id = reader_kernel_id;
    shared_variables.writer_kernel_id = writer_kernel_id;
    shared_variables.num_heads = num_heads;
    shared_variables.head_dim = head_dim;
    shared_variables.batch_size = BATCH_SIZE;
    shared_variables.cores = grid_to_cores(num_cores, grid_size.x, grid_size.y, true);

    cached_program_t cached_program{std::move(program), std::move(shared_variables)};
    override_runtime_arguments(cached_program, operation_attributes, tensor_args, tensor_return_value);
    return cached_program;
}

void GatedDeltaNetProgramFactory::override_runtime_arguments(
    cached_program_t& cached_program,
    const GatedDeltaNetParams& operation_attributes,
    const GatedDeltaNetInputs& tensor_args,
    std::vector<Tensor>& tensor_return_value) {
    auto& output = tensor_return_value[0];
    auto& new_state = tensor_return_value[1];
    auto& program = cached_program.program;
    const auto& sv = cached_program.shared_variables;

    uint32_t scaler_val = pack_bf16_pair(1.0f / static_cast<float>(sv.head_dim));
    uint32_t eps_val = pack_bf16_pair(operation_attributes.norm_eps);
    uint32_t scaler_one_val = pack_bf16_pair(1.0f);
    uint32_t q_scale_val = pack_bf16_pair(operation_attributes.scale);

    for (uint32_t i = 0; i < sv.cores.size(); i++) {
        SetRuntimeArgs(
            program,
            sv.reader_kernel_id,
            sv.cores[i],
            {tensor_args.conv_out.buffer()->address(),
             tensor_args.z_flat.buffer()->address(),
             tensor_args.ba_flat.buffer()->address(),
             tensor_args.dt_bias.buffer()->address(),
             tensor_args.neg_A_exp.buffer()->address(),
             tensor_args.state.buffer()->address(),
             tensor_args.norm_weight.buffer()->address(),
             i,
             1,
             scaler_val,
             eps_val,
             scaler_one_val,
             q_scale_val});

        SetRuntimeArgs(
            program,
            sv.writer_kernel_id,
            sv.cores[i],
            {output.buffer()->address(), new_state.buffer()->address(), i, 1});

        SetRuntimeArgs(program, sv.compute_kernel_id, sv.cores[i], {sv.batch_size});
    }
}

}  // namespace ttnn::experimental::prim

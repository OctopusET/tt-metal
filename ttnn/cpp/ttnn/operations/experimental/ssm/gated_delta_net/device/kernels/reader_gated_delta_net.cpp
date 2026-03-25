// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Reader for fused GatedDeltaNet kernel.
// Reads q/k/v from flat conv_out tensor using per-head tile offsets.
// Reads z from flat z_flat tensor. Extracts b/a scalars from ba_flat.
// Constructs k_col tiles via row-to-column byte transformation.

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "ttnn/cpp/ttnn/kernel/dataflow/generate_reduce_scaler.hpp"
#include "ttnn/cpp/ttnn/kernel/dataflow/generate_bcast_scalar.hpp"

constexpr uint32_t cb_q = get_compile_time_arg_val(0);
constexpr uint32_t cb_k_col = get_compile_time_arg_val(1);
constexpr uint32_t cb_v = get_compile_time_arg_val(2);
constexpr uint32_t cb_k_row = get_compile_time_arg_val(3);
constexpr uint32_t cb_b = get_compile_time_arg_val(4);
constexpr uint32_t cb_a = get_compile_time_arg_val(5);
constexpr uint32_t cb_dt_bias = get_compile_time_arg_val(6);
constexpr uint32_t cb_neg_A_exp = get_compile_time_arg_val(7);
constexpr uint32_t cb_state = get_compile_time_arg_val(8);
constexpr uint32_t D_TILES = get_compile_time_arg_val(9);
constexpr uint32_t STATE_TILES = get_compile_time_arg_val(10);
constexpr uint32_t cb_z = get_compile_time_arg_val(11);
constexpr uint32_t cb_norm_w = get_compile_time_arg_val(12);
constexpr uint32_t cb_scaler = get_compile_time_arg_val(13);
constexpr uint32_t cb_eps = get_compile_time_arg_val(14);
constexpr uint32_t cb_scaler_one = get_compile_time_arg_val(15);
constexpr uint32_t cb_q_scale = get_compile_time_arg_val(16);
constexpr uint32_t KEY_DIM_TILES = get_compile_time_arg_val(17);
constexpr uint32_t GQA_RATIO = get_compile_time_arg_val(18);

// TensorAccessors: conv_out, z_flat, ba_flat, dt_bias, neg_A_exp, state, norm_w
constexpr auto conv_out_acc_args = TensorAccessorArgs<19>();
constexpr auto z_flat_acc_args = TensorAccessorArgs<conv_out_acc_args.next_compile_time_args_offset()>();
constexpr auto ba_flat_acc_args = TensorAccessorArgs<z_flat_acc_args.next_compile_time_args_offset()>();
constexpr auto dt_bias_acc_args = TensorAccessorArgs<ba_flat_acc_args.next_compile_time_args_offset()>();
constexpr auto neg_A_exp_acc_args = TensorAccessorArgs<dt_bias_acc_args.next_compile_time_args_offset()>();
constexpr auto state_acc_args = TensorAccessorArgs<neg_A_exp_acc_args.next_compile_time_args_offset()>();
constexpr auto norm_w_acc_args = TensorAccessorArgs<state_acc_args.next_compile_time_args_offset()>();

void kernel_main() {
    uint32_t conv_out_addr = get_arg_val<uint32_t>(0);
    uint32_t z_flat_addr = get_arg_val<uint32_t>(1);
    uint32_t ba_flat_addr = get_arg_val<uint32_t>(2);
    uint32_t dt_bias_addr = get_arg_val<uint32_t>(3);
    uint32_t neg_A_exp_addr = get_arg_val<uint32_t>(4);
    uint32_t state_addr = get_arg_val<uint32_t>(5);
    uint32_t norm_w_addr = get_arg_val<uint32_t>(6);
    uint32_t head_start = get_arg_val<uint32_t>(7);
    uint32_t num_heads = get_arg_val<uint32_t>(8);
    uint32_t scaler_packed = get_arg_val<uint32_t>(9);
    uint32_t eps_packed = get_arg_val<uint32_t>(10);
    uint32_t scaler_one_packed = get_arg_val<uint32_t>(11);
    uint32_t q_scale_packed = get_arg_val<uint32_t>(12);

    const uint32_t bf16_tile_bytes = get_tile_size(cb_q);
    const uint32_t state_tile_bytes = get_tile_size(cb_state);

    const auto conv_out_acc = TensorAccessor(conv_out_acc_args, conv_out_addr, bf16_tile_bytes);
    const auto z_flat_acc = TensorAccessor(z_flat_acc_args, z_flat_addr, bf16_tile_bytes);
    const auto ba_flat_acc = TensorAccessor(ba_flat_acc_args, ba_flat_addr, bf16_tile_bytes);
    const auto dt_bias_acc = TensorAccessor(dt_bias_acc_args, dt_bias_addr, bf16_tile_bytes);
    const auto neg_A_exp_acc = TensorAccessor(neg_A_exp_acc_args, neg_A_exp_addr, bf16_tile_bytes);
    const auto state_acc = TensorAccessor(state_acc_args, state_addr, state_tile_bytes);
    const auto norm_w_acc = TensorAccessor(norm_w_acc_args, norm_w_addr, bf16_tile_bytes);

    // Generate constant tiles
    generate_reduce_scaler(cb_scaler, scaler_packed);
    generate_bcast_unary_scalar(cb_eps, eps_packed);
    generate_reduce_scaler(cb_scaler_one, scaler_one_packed);
    generate_bcast_unary_scalar(cb_q_scale, q_scale_packed);

    for (uint32_t hh = 0; hh < num_heads; hh++) {
        uint32_t h = head_start + hh;     // v_head index
        uint32_t k_head = h / GQA_RATIO;  // k_head index for GQA

        // conv_out layout: (1, 1, B, conv_dim) flat. Tiles are sequential.
        // q range: tiles [0, KEY_DIM_TILES). Head k_head: tiles k_head*D_TILES .. (k_head+1)*D_TILES-1
        // k range: tiles [KEY_DIM_TILES, 2*KEY_DIM_TILES). Head k_head: KEY_DIM_TILES + k_head*D_TILES ..
        // v range: tiles [2*KEY_DIM_TILES, ...). Head h: 2*KEY_DIM_TILES + h*D_TILES ..
        uint32_t q_base = k_head * D_TILES;
        uint32_t k_base = KEY_DIM_TILES + k_head * D_TILES;
        uint32_t v_base = 2 * KEY_DIM_TILES + h * D_TILES;

        // Read q from conv_out
        for (uint32_t t = 0; t < D_TILES; t++) {
            cb_reserve_back(cb_q, 1);
            noc_async_read_tile(q_base + t, conv_out_acc, get_write_ptr(cb_q));
            noc_async_read_barrier();
            cb_push_back(cb_q, 1);
        }

        // Read k_row + construct k_col
        for (uint32_t t = 0; t < D_TILES; t++) {
            cb_reserve_back(cb_k_row, 1);
            uint32_t k_row_addr = get_write_ptr(cb_k_row);
            noc_async_read_tile(k_base + t, conv_out_acc, k_row_addr);
            noc_async_read_barrier();
            cb_push_back(cb_k_row, 1);

            // Construct k_col: row 0 → col 0
            cb_reserve_back(cb_k_col, 1);
            uint32_t k_col_addr = get_write_ptr(cb_k_col);
            uint64_t zeros_noc_addr = get_noc_addr(MEM_ZEROS_BASE);
            constexpr uint32_t num_zeros_reads = 2048 / MEM_ZEROS_SIZE;
            uint32_t addr = k_col_addr;
            noc_async_read_one_packet_set_state(zeros_noc_addr, MEM_ZEROS_SIZE);
            for (uint32_t i = 0; i < num_zeros_reads; ++i) {
                noc_async_read_one_packet_with_state(zeros_noc_addr, addr);
                addr += MEM_ZEROS_SIZE;
            }
            noc_async_read_barrier();
            volatile tt_l1_ptr uint16_t* src = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(k_row_addr);
            volatile tt_l1_ptr uint16_t* dst = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(k_col_addr);
            for (uint32_t i = 0; i < 16; i++) {
                dst[i * 16] = src[i];
            }
            for (uint32_t i = 0; i < 16; i++) {
                dst[512 + i * 16] = src[256 + i];
            }
            cb_push_back(cb_k_col, 1);
        }

        // Read v from conv_out
        for (uint32_t t = 0; t < D_TILES; t++) {
            cb_reserve_back(cb_v, 1);
            noc_async_read_tile(v_base + t, conv_out_acc, get_write_ptr(cb_v));
            noc_async_read_barrier();
            cb_push_back(cb_v, 1);
        }

        // Extract b[h] from ba_flat: tile 0 has b[0..31] in row 0
        {
            cb_reserve_back(cb_b, 1);
            uint32_t tmp = get_write_ptr(cb_b);
            noc_async_read_tile(0, ba_flat_acc, tmp);
            noc_async_read_barrier();
            volatile tt_l1_ptr uint16_t* bptr = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(tmp);
            uint16_t b_val = (h < 16) ? bptr[h] : bptr[256 + (h - 16)];
            // Zero-fill and write scalar
            uint64_t zeros_noc_addr = get_noc_addr(MEM_ZEROS_BASE);
            constexpr uint32_t num_zeros_reads = 2048 / MEM_ZEROS_SIZE;
            uint32_t addr = tmp;
            noc_async_read_one_packet_set_state(zeros_noc_addr, MEM_ZEROS_SIZE);
            for (uint32_t i = 0; i < num_zeros_reads; ++i) {
                noc_async_read_one_packet_with_state(zeros_noc_addr, addr);
                addr += MEM_ZEROS_SIZE;
            }
            noc_async_read_barrier();
            bptr[0] = b_val;
            cb_push_back(cb_b, 1);
        }

        // Extract a[h] from ba_flat: tile 1 has a[0..31] in row 0
        {
            cb_reserve_back(cb_a, 1);
            uint32_t tmp = get_write_ptr(cb_a);
            noc_async_read_tile(1, ba_flat_acc, tmp);
            noc_async_read_barrier();
            volatile tt_l1_ptr uint16_t* aptr = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(tmp);
            uint16_t a_val = (h < 16) ? aptr[h] : aptr[256 + (h - 16)];
            uint64_t zeros_noc_addr = get_noc_addr(MEM_ZEROS_BASE);
            constexpr uint32_t num_zeros_reads = 2048 / MEM_ZEROS_SIZE;
            uint32_t addr = tmp;
            noc_async_read_one_packet_set_state(zeros_noc_addr, MEM_ZEROS_SIZE);
            for (uint32_t i = 0; i < num_zeros_reads; ++i) {
                noc_async_read_one_packet_with_state(zeros_noc_addr, addr);
                addr += MEM_ZEROS_SIZE;
            }
            noc_async_read_barrier();
            aptr[0] = a_val;
            cb_push_back(cb_a, 1);
        }

        // Read dt_bias, neg_A_exp (per v_head)
        cb_reserve_back(cb_dt_bias, 1);
        noc_async_read_tile(h, dt_bias_acc, get_write_ptr(cb_dt_bias));
        noc_async_read_barrier();
        cb_push_back(cb_dt_bias, 1);

        cb_reserve_back(cb_neg_A_exp, 1);
        noc_async_read_tile(h, neg_A_exp_acc, get_write_ptr(cb_neg_A_exp));
        noc_async_read_barrier();
        cb_push_back(cb_neg_A_exp, 1);

        // Read state (per v_head)
        for (uint32_t t = 0; t < STATE_TILES; t++) {
            cb_reserve_back(cb_state, 1);
            noc_async_read_tile(h * STATE_TILES + t, state_acc, get_write_ptr(cb_state));
            noc_async_read_barrier();
            cb_push_back(cb_state, 1);
        }

        // Read z from z_flat (per v_head)
        for (uint32_t t = 0; t < D_TILES; t++) {
            cb_reserve_back(cb_z, 1);
            noc_async_read_tile(h * D_TILES + t, z_flat_acc, get_write_ptr(cb_z));
            noc_async_read_barrier();
            cb_push_back(cb_z, 1);
        }

        // Read norm_weight (per v_head)
        for (uint32_t t = 0; t < D_TILES; t++) {
            cb_reserve_back(cb_norm_w, 1);
            noc_async_read_tile(h * D_TILES + t, norm_w_acc, get_write_ptr(cb_norm_w));
            noc_async_read_barrier();
            cb_push_back(cb_norm_w, 1);
        }
    }
}

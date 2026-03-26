// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Reader for fused GatedDeltaNet kernel with batch support.
// For each head, loops over BATCH_SIZE batch items.
// Extracts row b from input tiles (placing into row 0) so compute processes one batch item at a time.

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
constexpr uint32_t BATCH_SIZE = get_compile_time_arg_val(19);
constexpr uint32_t STATE_BATCH_STRIDE = get_compile_time_arg_val(20);

// TensorAccessors start at index 21
constexpr auto conv_out_acc_args = TensorAccessorArgs<21>();
constexpr auto z_flat_acc_args = TensorAccessorArgs<conv_out_acc_args.next_compile_time_args_offset()>();
constexpr auto ba_flat_acc_args = TensorAccessorArgs<z_flat_acc_args.next_compile_time_args_offset()>();
constexpr auto dt_bias_acc_args = TensorAccessorArgs<ba_flat_acc_args.next_compile_time_args_offset()>();
constexpr auto neg_A_exp_acc_args = TensorAccessorArgs<dt_bias_acc_args.next_compile_time_args_offset()>();
constexpr auto state_acc_args = TensorAccessorArgs<neg_A_exp_acc_args.next_compile_time_args_offset()>();
constexpr auto norm_w_acc_args = TensorAccessorArgs<state_acc_args.next_compile_time_args_offset()>();

// Zero-fill a bf16 tile in L1 (via direct store, not NOC read)
inline void zero_fill_tile(uint32_t tile_addr) {
    volatile tt_l1_ptr uint32_t* p = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(tile_addr);
    for (uint32_t i = 0; i < 512; i++) {  // 2048 bytes / 4
        p[i] = 0;
    }
}

// Get bf16 tile face offsets for row b (in uint16_t units).
// Returns (face0_or_2_offset, face1_or_3_offset), each pointing to start of 16-element row.
inline void get_row_offsets(uint32_t b, uint32_t& off0, uint32_t& off1) {
    if (b < 16) {
        off0 = b * 16;        // face 0
        off1 = 256 + b * 16;  // face 1
    } else {
        off0 = 512 + (b - 16) * 16;  // face 2
        off1 = 768 + (b - 16) * 16;  // face 3
    }
}

// Get bf16 element index for (row, col) in tile.
inline uint32_t tile_elem_idx(uint32_t row, uint32_t col) {
    uint32_t face_row = row & 15;
    uint32_t face_col = col & 15;
    uint32_t face_idx = (row >= 16 ? 2 : 0) + (col >= 16 ? 1 : 0);
    return face_idx * 256 + face_row * 16 + face_col;
}

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

    for (uint32_t hh = 0; hh < num_heads; hh++) {
        uint32_t h = head_start + hh;
        uint32_t k_head = h / GQA_RATIO;

        uint32_t q_base = k_head * D_TILES;
        uint32_t k_base = KEY_DIM_TILES + k_head * D_TILES;
        uint32_t v_base = 2 * KEY_DIM_TILES + h * D_TILES;

        // Push scaler constants once per head (compute reuses without popping)
        generate_reduce_scaler(cb_scaler, scaler_packed);
        generate_bcast_unary_scalar(cb_eps, eps_packed);
        generate_reduce_scaler(cb_scaler_one, scaler_one_packed);
        generate_bcast_unary_scalar(cb_q_scale, q_scale_packed);

        for (uint32_t bi = 0; bi < BATCH_SIZE; bi++) {
            uint32_t r0, r1;  // row offsets for batch item bi
            get_row_offsets(bi, r0, r1);

            // --- Q tiles: read from DRAM, extract row bi to row 0 ---
            for (uint32_t t = 0; t < D_TILES; t++) {
                cb_reserve_back(cb_q, 1);
                uint32_t addr = get_write_ptr(cb_q);
                noc_async_read_tile(q_base + t, conv_out_acc, addr);
                noc_async_read_barrier();
                if (bi > 0) {
                    volatile tt_l1_ptr uint16_t* p = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(addr);
                    for (uint32_t i = 0; i < 16; i++) {
                        p[i] = p[r0 + i];
                        p[256 + i] = p[r1 + i];
                    }
                }
                cb_push_back(cb_q, 1);
            }

            // --- K tiles: read, extract row bi, construct k_row + k_col ---
            for (uint32_t t = 0; t < D_TILES; t++) {
                cb_reserve_back(cb_k_row, 1);
                uint32_t k_addr = get_write_ptr(cb_k_row);
                noc_async_read_tile(k_base + t, conv_out_acc, k_addr);
                noc_async_read_barrier();

                volatile tt_l1_ptr uint16_t* ks = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(k_addr);
                // Save row bi values for k_col construction
                uint16_t kv[32];
                for (uint32_t i = 0; i < 16; i++) {
                    kv[i] = ks[r0 + i];
                    kv[16 + i] = ks[r1 + i];
                }
                // k_row: copy row bi to row 0
                if (bi > 0) {
                    for (uint32_t i = 0; i < 16; i++) {
                        ks[i] = kv[i];
                        ks[256 + i] = kv[16 + i];
                    }
                }
                cb_push_back(cb_k_row, 1);

                // k_col: zero-fill, write k values to column 0
                cb_reserve_back(cb_k_col, 1);
                uint32_t kc_addr = get_write_ptr(cb_k_col);
                zero_fill_tile(kc_addr);
                volatile tt_l1_ptr uint16_t* kc = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(kc_addr);
                for (uint32_t i = 0; i < 16; i++) {
                    kc[i * 16] = kv[i];             // face 0 col 0
                    kc[512 + i * 16] = kv[16 + i];  // face 2 col 0
                }
                cb_push_back(cb_k_col, 1);
            }

            // --- V tiles: read, extract row bi to row 0 ---
            for (uint32_t t = 0; t < D_TILES; t++) {
                cb_reserve_back(cb_v, 1);
                uint32_t addr = get_write_ptr(cb_v);
                noc_async_read_tile(v_base + t, conv_out_acc, addr);
                noc_async_read_barrier();
                if (bi > 0) {
                    volatile tt_l1_ptr uint16_t* p = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(addr);
                    for (uint32_t i = 0; i < 16; i++) {
                        p[i] = p[r0 + i];
                        p[256 + i] = p[r1 + i];
                    }
                }
                cb_push_back(cb_v, 1);
            }

            // --- b scalar: extract from row bi, col h of ba_flat tile 0 ---
            {
                cb_reserve_back(cb_b, 1);
                uint32_t tmp = get_write_ptr(cb_b);
                noc_async_read_tile(0, ba_flat_acc, tmp);
                noc_async_read_barrier();
                volatile tt_l1_ptr uint16_t* bp = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(tmp);
                uint16_t b_val = bp[tile_elem_idx(bi, h)];
                zero_fill_tile(tmp);
                bp[0] = b_val;
                cb_push_back(cb_b, 1);
            }

            // --- a scalar: extract from row bi, col h of ba_flat tile 1 ---
            {
                cb_reserve_back(cb_a, 1);
                uint32_t tmp = get_write_ptr(cb_a);
                noc_async_read_tile(1, ba_flat_acc, tmp);
                noc_async_read_barrier();
                volatile tt_l1_ptr uint16_t* ap = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(tmp);
                uint16_t a_val = ap[tile_elem_idx(bi, h)];
                zero_fill_tile(tmp);
                ap[0] = a_val;
                cb_push_back(cb_a, 1);
            }

            // --- dt_bias, neg_A_exp (per head, same for all batch items) ---
            cb_reserve_back(cb_dt_bias, 1);
            noc_async_read_tile(h, dt_bias_acc, get_write_ptr(cb_dt_bias));
            noc_async_read_barrier();
            cb_push_back(cb_dt_bias, 1);

            cb_reserve_back(cb_neg_A_exp, 1);
            noc_async_read_tile(h, neg_A_exp_acc, get_write_ptr(cb_neg_A_exp));
            noc_async_read_barrier();
            cb_push_back(cb_neg_A_exp, 1);

            // --- State for batch item bi ---
            for (uint32_t t = 0; t < STATE_TILES; t++) {
                cb_reserve_back(cb_state, 1);
                noc_async_read_tile(bi * STATE_BATCH_STRIDE + h * STATE_TILES + t, state_acc, get_write_ptr(cb_state));
                noc_async_read_barrier();
                cb_push_back(cb_state, 1);
            }

            // --- Z tiles: read, extract row bi to row 0 ---
            for (uint32_t t = 0; t < D_TILES; t++) {
                cb_reserve_back(cb_z, 1);
                uint32_t addr = get_write_ptr(cb_z);
                noc_async_read_tile(h * D_TILES + t, z_flat_acc, addr);
                noc_async_read_barrier();
                if (bi > 0) {
                    volatile tt_l1_ptr uint16_t* p = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(addr);
                    for (uint32_t i = 0; i < 16; i++) {
                        p[i] = p[r0 + i];
                        p[256 + i] = p[r1 + i];
                    }
                }
                cb_push_back(cb_z, 1);
            }

            // --- norm_w (per head, same for all batch items) ---
            for (uint32_t t = 0; t < D_TILES; t++) {
                cb_reserve_back(cb_norm_w, 1);
                noc_async_read_tile(h * D_TILES + t, norm_w_acc, get_write_ptr(cb_norm_w));
                noc_async_read_barrier();
                cb_push_back(cb_norm_w, 1);
            }

            // Scaler constants pushed once per head, reused across batch iterations
        }
    }
}

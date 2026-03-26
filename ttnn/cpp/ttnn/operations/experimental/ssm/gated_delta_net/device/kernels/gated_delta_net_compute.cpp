// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// GatedDeltaNet fused kernel: L2norm + gates + retrieve + delta + recurrence + gated RMSNorm.
//
// Phase L: L2 normalize q and k (reduce + rsqrt + scalar bcast)
// Phase G: Compute gates (beta, decay) from raw inputs via SFPU fp32
// Phase 1: Decay state (fp32 state -> bf16 -> scalar bcast -> fp32 + bf16)
// Phase R: Retrieve kv_mem = k_row_norm @ decayed_state
// Phase D: Delta = (v - kv_mem) * beta
// Phase 2: Outer product k_col_norm @ delta -> [D,D]
// Phase 3: State update = decayed + outer -> fp32 (dual pack)
// Phase 4: Output = q_norm @ state_updated -> cb_tmp (raw, for RMSNorm)
// Phase 5: Gated RMSNorm = output * rsqrt(mean(x^2)+eps) * norm_w * silu(z)

#include <cstdint>

#define REDUCE_OP PoolType::SUM
#define REDUCE_DIM ReduceDim::REDUCE_ROW

#include "api/compute/eltwise_binary.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/compute/eltwise_unary/exp.h"
#include "api/compute/eltwise_unary/softplus.h"
#include "api/compute/eltwise_unary/rsqrt.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/compute_kernel_api.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/bcast.h"
#include "api/compute/matmul.h"
#include "api/compute/pack.h"
#include "api/compute/reduce.h"

constexpr uint32_t cb_q = get_compile_time_arg_val(0);
constexpr uint32_t cb_k_col = get_compile_time_arg_val(1);
constexpr uint32_t cb_v = get_compile_time_arg_val(2);
constexpr uint32_t cb_decay = get_compile_time_arg_val(3);
constexpr uint32_t cb_beta = get_compile_time_arg_val(4);
constexpr uint32_t cb_state = get_compile_time_arg_val(5);
constexpr uint32_t cb_state_new = get_compile_time_arg_val(6);
constexpr uint32_t cb_out = get_compile_time_arg_val(7);
constexpr uint32_t cb_sd = get_compile_time_arg_val(8);
constexpr uint32_t cb_tmp = get_compile_time_arg_val(9);
constexpr uint32_t cb_tmp2 = get_compile_time_arg_val(10);
constexpr uint32_t cb_sd2 = get_compile_time_arg_val(11);
constexpr uint32_t D_TILES = get_compile_time_arg_val(12);
constexpr uint32_t STATE_TILES = get_compile_time_arg_val(13);
constexpr uint32_t cb_k_row = get_compile_time_arg_val(14);
constexpr uint32_t cb_b = get_compile_time_arg_val(15);
constexpr uint32_t cb_a = get_compile_time_arg_val(16);
constexpr uint32_t cb_dt_bias = get_compile_time_arg_val(17);
constexpr uint32_t cb_neg_A_exp = get_compile_time_arg_val(18);
constexpr uint32_t cb_kv_mem = get_compile_time_arg_val(19);
constexpr uint32_t cb_delta = get_compile_time_arg_val(20);
constexpr uint32_t cb_z = get_compile_time_arg_val(21);
constexpr uint32_t cb_norm_w = get_compile_time_arg_val(22);
constexpr uint32_t cb_scaler = get_compile_time_arg_val(23);
constexpr uint32_t cb_eps = get_compile_time_arg_val(24);
constexpr uint32_t cb_q_norm = get_compile_time_arg_val(25);
constexpr uint32_t cb_k_row_norm = get_compile_time_arg_val(26);
constexpr uint32_t cb_k_col_norm = get_compile_time_arg_val(27);
constexpr uint32_t cb_scaler_one = get_compile_time_arg_val(28);
constexpr uint32_t cb_q_scale = get_compile_time_arg_val(29);

constexpr uint32_t SP_BETA = 0x3F800000u;
constexpr uint32_t SP_BETA_RECIP = 0x3F800000u;
constexpr uint32_t SP_THRESHOLD = 0x41A00000u;

void kernel_main() {
    const uint32_t num_heads_this_core = get_arg_val<uint32_t>(0);

    for (uint32_t hh = 0; hh < num_heads_this_core; hh++) {
        cb_wait_front(cb_q, D_TILES);
        cb_wait_front(cb_k_col, D_TILES);
        cb_wait_front(cb_k_row, D_TILES);
        cb_wait_front(cb_v, D_TILES);
        cb_wait_front(cb_b, 1);
        cb_wait_front(cb_a, 1);
        cb_wait_front(cb_dt_bias, 1);
        cb_wait_front(cb_neg_A_exp, 1);
        cb_wait_front(cb_state, STATE_TILES);
        cb_wait_front(cb_z, D_TILES);
        cb_wait_front(cb_norm_w, D_TILES);

        // ============================================================
        // Phase L1: q L2 normalization -> cb_q_norm
        // q_norm = q / ||q|| * scale
        // ============================================================

        // L1a: Square q tiles -> cb_tmp2
        cb_reserve_back(cb_tmp2, D_TILES);
        init_sfpu(cb_q, cb_tmp2);
        for (uint32_t t = 0; t < D_TILES; t++) {
            tile_regs_acquire();
            copy_tile(cb_q, t, 0);
            mul_binary_tile_init();
            mul_binary_tile(0, 0, 0);
            tile_regs_commit();
            tile_regs_wait();
            pack_tile(0, cb_tmp2);
            tile_regs_release();
        }
        cb_push_back(cb_tmp2, D_TILES);

        // L1b: Reduce sum(q^2) with scaler=1.0
        cb_wait_front(cb_tmp2, D_TILES);
        cb_wait_front(cb_scaler_one, 1);
        cb_reserve_back(cb_kv_mem, 1);
        reduce_init<PoolType::SUM, ReduceDim::REDUCE_ROW, true>(cb_tmp2, cb_scaler_one, cb_kv_mem);
        tile_regs_acquire();
        for (uint32_t t = 0; t < D_TILES; t++) {
            reduce_tile<PoolType::SUM, ReduceDim::REDUCE_ROW, true>(cb_tmp2, cb_scaler_one, t, 0, 0);
        }
        tile_regs_commit();
        tile_regs_wait();
        pack_tile(0, cb_kv_mem);
        tile_regs_release();
        cb_push_back(cb_kv_mem, 1);
        cb_pop_front(cb_tmp2, D_TILES);
        reduce_uninit<true>(cb_tmp2);

        // L1c: inv_norm_q = rsqrt(sum_sq) * scale
        cb_wait_front(cb_kv_mem, 1);
        cb_wait_front(cb_q_scale, 1);
        init_sfpu(cb_kv_mem, cb_kv_mem);
        tile_regs_acquire();
        copy_tile(cb_kv_mem, 0, 0);
        rsqrt_tile_init();
        rsqrt_tile(0);
        copy_tile(cb_q_scale, 0, 1);
        mul_binary_tile_init();
        mul_binary_tile(0, 1, 0);
        tile_regs_commit();
        tile_regs_wait();
        cb_pop_front(cb_kv_mem, 1);
        cb_reserve_back(cb_kv_mem, 1);
        pack_tile(0, cb_kv_mem);
        tile_regs_release();
        cb_push_back(cb_kv_mem, 1);

        // L1d: q * inv_norm_q_scaled -> cb_q_norm
        cb_wait_front(cb_kv_mem, 1);
        cb_reserve_back(cb_q_norm, D_TILES);
        binary_op_init_common(cb_q, cb_kv_mem, cb_q_norm);
        mul_tiles_bcast_scalar_init_short(cb_q, cb_kv_mem);
        for (uint32_t t = 0; t < D_TILES; t++) {
            tile_regs_acquire();
            mul_tiles_bcast_scalar(cb_q, cb_kv_mem, t, 0, 0);
            tile_regs_commit();
            tile_regs_wait();
            pack_tile(0, cb_q_norm);
            tile_regs_release();
        }
        cb_push_back(cb_q_norm, D_TILES);
        cb_pop_front(cb_q, D_TILES);
        cb_pop_front(cb_kv_mem, 1);

        // ============================================================
        // Phase L2: k L2 normalization -> cb_k_row_norm, cb_k_col_norm
        // k_norm = k / ||k||  (no scale for k)
        // ============================================================

        // L2a: Square k_row tiles -> cb_tmp2
        cb_reserve_back(cb_tmp2, D_TILES);
        init_sfpu(cb_k_row, cb_tmp2);
        for (uint32_t t = 0; t < D_TILES; t++) {
            tile_regs_acquire();
            copy_tile(cb_k_row, t, 0);
            mul_binary_tile_init();
            mul_binary_tile(0, 0, 0);
            tile_regs_commit();
            tile_regs_wait();
            pack_tile(0, cb_tmp2);
            tile_regs_release();
        }
        cb_push_back(cb_tmp2, D_TILES);

        // L2b: Reduce sum(k^2)
        cb_wait_front(cb_tmp2, D_TILES);
        cb_reserve_back(cb_kv_mem, 1);
        reduce_init<PoolType::SUM, ReduceDim::REDUCE_ROW, true>(cb_tmp2, cb_scaler_one, cb_kv_mem);
        tile_regs_acquire();
        for (uint32_t t = 0; t < D_TILES; t++) {
            reduce_tile<PoolType::SUM, ReduceDim::REDUCE_ROW, true>(cb_tmp2, cb_scaler_one, t, 0, 0);
        }
        tile_regs_commit();
        tile_regs_wait();
        pack_tile(0, cb_kv_mem);
        tile_regs_release();
        cb_push_back(cb_kv_mem, 1);
        cb_pop_front(cb_tmp2, D_TILES);
        reduce_uninit<true>(cb_tmp2);

        // L2c: inv_norm_k = rsqrt(sum_sq)
        cb_wait_front(cb_kv_mem, 1);
        init_sfpu(cb_kv_mem, cb_kv_mem);
        tile_regs_acquire();
        copy_tile(cb_kv_mem, 0, 0);
        rsqrt_tile_init();
        rsqrt_tile(0);
        tile_regs_commit();
        tile_regs_wait();
        cb_pop_front(cb_kv_mem, 1);
        cb_reserve_back(cb_kv_mem, 1);
        pack_tile(0, cb_kv_mem);
        tile_regs_release();
        cb_push_back(cb_kv_mem, 1);

        // L2d: k_row * inv_norm_k -> cb_k_row_norm
        cb_wait_front(cb_kv_mem, 1);
        cb_reserve_back(cb_k_row_norm, D_TILES);
        binary_op_init_common(cb_k_row, cb_kv_mem, cb_k_row_norm);
        mul_tiles_bcast_scalar_init_short(cb_k_row, cb_kv_mem);
        for (uint32_t t = 0; t < D_TILES; t++) {
            tile_regs_acquire();
            mul_tiles_bcast_scalar(cb_k_row, cb_kv_mem, t, 0, 0);
            tile_regs_commit();
            tile_regs_wait();
            pack_tile(0, cb_k_row_norm);
            tile_regs_release();
        }
        cb_push_back(cb_k_row_norm, D_TILES);
        cb_pop_front(cb_k_row, D_TILES);

        // L2e: k_col * inv_norm_k -> cb_k_col_norm (same inv_norm!)
        cb_reserve_back(cb_k_col_norm, D_TILES);
        binary_op_init_common(cb_k_col, cb_kv_mem, cb_k_col_norm);
        mul_tiles_bcast_scalar_init_short(cb_k_col, cb_kv_mem);
        for (uint32_t t = 0; t < D_TILES; t++) {
            tile_regs_acquire();
            mul_tiles_bcast_scalar(cb_k_col, cb_kv_mem, t, 0, 0);
            tile_regs_commit();
            tile_regs_wait();
            pack_tile(0, cb_k_col_norm);
            tile_regs_release();
        }
        cb_push_back(cb_k_col_norm, D_TILES);
        cb_pop_front(cb_k_col, D_TILES);
        cb_pop_front(cb_kv_mem, 1);

        // ============================================================
        // Phase G1: beta = sigmoid(b)
        // ============================================================
        cb_reserve_back(cb_beta, 1);
        init_sfpu(cb_b, cb_beta);
        tile_regs_acquire();
        copy_tile(cb_b, 0, 0);
        sigmoid_tile_init();
        sigmoid_tile(0);
        tile_regs_commit();
        tile_regs_wait();
        pack_tile(0, cb_beta);
        tile_regs_release();
        cb_push_back(cb_beta, 1);
        cb_pop_front(cb_b, 1);

        // ============================================================
        // Phase G2: decay = exp(neg_A_exp * softplus(a + dt_bias))
        // ============================================================
        cb_reserve_back(cb_decay, 1);
        init_sfpu(cb_a, cb_decay);
        tile_regs_acquire();
        copy_tile(cb_a, 0, 0);
        copy_tile(cb_dt_bias, 0, 1);
        add_binary_tile_init();
        add_binary_tile(0, 1, 0);
        softplus_tile_init();
        softplus_tile(0, SP_BETA, SP_BETA_RECIP, SP_THRESHOLD);
        copy_tile(cb_neg_A_exp, 0, 1);
        mul_binary_tile_init();
        mul_binary_tile(0, 1, 0);
        exp_tile_init();
        exp_tile(0);
        tile_regs_commit();
        tile_regs_wait();
        pack_tile(0, cb_decay);
        tile_regs_release();
        cb_push_back(cb_decay, 1);
        cb_pop_front(cb_a, 1);
        cb_pop_front(cb_dt_bias, 1);
        cb_pop_front(cb_neg_A_exp, 1);

        // ============================================================
        // Phase 1a: fp32 state -> bf16 cb_sd
        // ============================================================
        cb_reserve_back(cb_sd, STATE_TILES);
        init_sfpu(cb_state, cb_sd);
        for (uint32_t t = 0; t < STATE_TILES; t++) {
            tile_regs_acquire();
            copy_tile(cb_state, t, 0);
            tile_regs_commit();
            tile_regs_wait();
            pack_tile(0, cb_sd);
            tile_regs_release();
        }
        cb_push_back(cb_sd, STATE_TILES);
        cb_pop_front(cb_state, STATE_TILES);

        // ============================================================
        // Phase 1b: decay * state -> fp32 cb_state + bf16 cb_sd2
        // ============================================================
        cb_wait_front(cb_sd, STATE_TILES);
        cb_wait_front(cb_decay, 1);
        cb_reserve_back(cb_state, STATE_TILES);
        cb_reserve_back(cb_sd2, STATE_TILES);
        binary_op_init_common(cb_sd, cb_decay, cb_state);
        mul_tiles_bcast_scalar_init_short(cb_sd, cb_decay);
        for (uint32_t t = 0; t < STATE_TILES; t++) {
            tile_regs_acquire();
            mul_tiles_bcast_scalar(cb_sd, cb_decay, t, 0, 0);
            tile_regs_commit();
            tile_regs_wait();
            pack_tile(0, cb_state);
            pack_reconfig_data_format(cb_sd2);
            pack_tile(0, cb_sd2);
            pack_reconfig_data_format(cb_state);
            tile_regs_release();
        }
        cb_push_back(cb_state, STATE_TILES);
        cb_push_back(cb_sd2, STATE_TILES);
        cb_pop_front(cb_sd, STATE_TILES);
        cb_pop_front(cb_decay, 1);

        // ============================================================
        // Phase R: kv_mem = k_row_norm @ state_decayed
        // ============================================================
        cb_wait_front(cb_k_row_norm, D_TILES);
        cb_reserve_back(cb_kv_mem, D_TILES);
        mm_init(cb_k_row_norm, cb_state, cb_kv_mem);
        for (uint32_t j = 0; j < D_TILES; j++) {
            tile_regs_acquire();
            for (uint32_t i = 0; i < D_TILES; i++) {
                matmul_tiles(cb_k_row_norm, cb_state, i, i * D_TILES + j, 0);
            }
            tile_regs_commit();
            tile_regs_wait();
            pack_tile(0, cb_kv_mem);
            tile_regs_release();
        }
        cb_push_back(cb_kv_mem, D_TILES);
        cb_pop_front(cb_k_row_norm, D_TILES);

        // ============================================================
        // Phase D: delta = (v - kv_mem) * beta
        // ============================================================
        cb_wait_front(cb_kv_mem, D_TILES);

        cb_reserve_back(cb_tmp, D_TILES);
        binary_op_init_common(cb_v, cb_kv_mem, cb_tmp);
        sub_tiles_init(cb_v, cb_kv_mem);
        for (uint32_t t = 0; t < D_TILES; t++) {
            tile_regs_acquire();
            sub_tiles(cb_v, cb_kv_mem, t, t, 0);
            tile_regs_commit();
            tile_regs_wait();
            pack_tile(0, cb_tmp);
            tile_regs_release();
        }
        cb_push_back(cb_tmp, D_TILES);
        cb_pop_front(cb_v, D_TILES);
        cb_pop_front(cb_kv_mem, D_TILES);

        cb_wait_front(cb_tmp, D_TILES);
        cb_wait_front(cb_beta, 1);
        cb_reserve_back(cb_delta, D_TILES);
        binary_op_init_common(cb_tmp, cb_beta, cb_delta);
        mul_tiles_bcast_scalar_init_short(cb_tmp, cb_beta);
        for (uint32_t t = 0; t < D_TILES; t++) {
            tile_regs_acquire();
            mul_tiles_bcast_scalar(cb_tmp, cb_beta, t, 0, 0);
            tile_regs_commit();
            tile_regs_wait();
            pack_tile(0, cb_delta);
            tile_regs_release();
        }
        cb_push_back(cb_delta, D_TILES);
        cb_pop_front(cb_tmp, D_TILES);
        cb_pop_front(cb_beta, 1);

        // ============================================================
        // Phase 2: outer product k_col_norm @ delta
        // ============================================================
        cb_wait_front(cb_k_col_norm, D_TILES);
        cb_wait_front(cb_delta, D_TILES);
        cb_reserve_back(cb_sd, STATE_TILES);
        mm_init(cb_k_col_norm, cb_delta, cb_sd);
        for (uint32_t i = 0; i < D_TILES; i++) {
            for (uint32_t j = 0; j < D_TILES; j++) {
                tile_regs_acquire();
                matmul_tiles(cb_k_col_norm, cb_delta, i, j, 0);
                tile_regs_commit();
                tile_regs_wait();
                pack_tile(0, cb_sd);
                tile_regs_release();
            }
        }
        cb_push_back(cb_sd, STATE_TILES);

        // ============================================================
        // Phase 3: state_new = decayed + outer
        // ============================================================
        cb_wait_front(cb_sd, STATE_TILES);
        cb_wait_front(cb_sd2, STATE_TILES);
        cb_pop_front(cb_state, STATE_TILES);
        cb_reserve_back(cb_state, STATE_TILES);
        cb_reserve_back(cb_state_new, STATE_TILES);

        binary_op_init_common(cb_sd2, cb_sd, cb_state);
        add_tiles_init(cb_sd2, cb_sd);
        for (uint32_t t = 0; t < STATE_TILES; t++) {
            tile_regs_acquire();
            add_tiles(cb_sd2, cb_sd, t, t, 0);
            tile_regs_commit();
            tile_regs_wait();
            pack_tile(0, cb_state);
            pack_tile(0, cb_state_new);
            tile_regs_release();
        }
        cb_push_back(cb_state, STATE_TILES);
        cb_push_back(cb_state_new, STATE_TILES);
        cb_pop_front(cb_sd, STATE_TILES);
        cb_pop_front(cb_sd2, STATE_TILES);

        // ============================================================
        // Phase 4: raw_output = q_norm @ state_updated -> cb_tmp
        // ============================================================
        cb_wait_front(cb_q_norm, D_TILES);
        cb_wait_front(cb_state, STATE_TILES);
        cb_reserve_back(cb_tmp, D_TILES);
        mm_init(cb_q_norm, cb_state, cb_tmp);
        for (uint32_t j = 0; j < D_TILES; j++) {
            tile_regs_acquire();
            for (uint32_t i = 0; i < D_TILES; i++) {
                matmul_tiles(cb_q_norm, cb_state, i, i * D_TILES + j, 0);
            }
            tile_regs_commit();
            tile_regs_wait();
            pack_tile(0, cb_tmp);
            tile_regs_release();
        }
        cb_push_back(cb_tmp, D_TILES);
        cb_pop_front(cb_state, STATE_TILES);
        cb_pop_front(cb_q_norm, D_TILES);
        cb_pop_front(cb_k_col_norm, D_TILES);
        cb_pop_front(cb_delta, D_TILES);

        // ============================================================
        // Phase 5: Gated RMSNorm
        // output_final = raw_output * rsqrt(mean(x^2) + eps) * norm_w * silu(z)
        // ============================================================

        // 5a: Square raw output tiles -> cb_tmp2
        cb_wait_front(cb_tmp, D_TILES);
        cb_reserve_back(cb_tmp2, D_TILES);
        init_sfpu(cb_tmp, cb_tmp2);
        for (uint32_t t = 0; t < D_TILES; t++) {
            tile_regs_acquire();
            copy_tile(cb_tmp, t, 0);
            mul_binary_tile_init();
            mul_binary_tile(0, 0, 0);
            tile_regs_commit();
            tile_regs_wait();
            pack_tile(0, cb_tmp2);
            tile_regs_release();
        }
        cb_push_back(cb_tmp2, D_TILES);

        // 5b: Reduce sum(x^2) with scaler = 1/D -> variance
        cb_wait_front(cb_tmp2, D_TILES);
        cb_wait_front(cb_scaler, 1);
        cb_reserve_back(cb_kv_mem, 1);
        reduce_init<PoolType::SUM, ReduceDim::REDUCE_ROW, true>(cb_tmp2, cb_scaler, cb_kv_mem);
        tile_regs_acquire();
        for (uint32_t t = 0; t < D_TILES; t++) {
            reduce_tile<PoolType::SUM, ReduceDim::REDUCE_ROW, true>(cb_tmp2, cb_scaler, t, 0, 0);
        }
        tile_regs_commit();
        tile_regs_wait();
        pack_tile(0, cb_kv_mem);
        tile_regs_release();
        cb_push_back(cb_kv_mem, 1);
        cb_pop_front(cb_tmp2, D_TILES);
        reduce_uninit<true>(cb_tmp2);

        // 5c: inv_std = rsqrt(variance + eps)
        cb_wait_front(cb_kv_mem, 1);
        cb_wait_front(cb_eps, 1);
        init_sfpu(cb_kv_mem, cb_kv_mem);
        tile_regs_acquire();
        copy_tile(cb_kv_mem, 0, 0);
        copy_tile(cb_eps, 0, 1);
        add_binary_tile_init();
        add_binary_tile(0, 1, 0);
        rsqrt_tile_init();
        rsqrt_tile(0);
        tile_regs_commit();
        tile_regs_wait();
        cb_pop_front(cb_kv_mem, 1);
        cb_reserve_back(cb_kv_mem, 1);
        pack_tile(0, cb_kv_mem);
        tile_regs_release();
        cb_push_back(cb_kv_mem, 1);

        // 5d: raw_output * inv_std -> cb_tmp2
        cb_wait_front(cb_kv_mem, 1);
        cb_reserve_back(cb_tmp2, D_TILES);
        binary_op_init_common(cb_tmp, cb_kv_mem, cb_tmp2);
        mul_tiles_bcast_scalar_init_short(cb_tmp, cb_kv_mem);
        for (uint32_t t = 0; t < D_TILES; t++) {
            tile_regs_acquire();
            mul_tiles_bcast_scalar(cb_tmp, cb_kv_mem, t, 0, 0);
            tile_regs_commit();
            tile_regs_wait();
            pack_tile(0, cb_tmp2);
            tile_regs_release();
        }
        cb_push_back(cb_tmp2, D_TILES);
        cb_pop_front(cb_tmp, D_TILES);
        cb_pop_front(cb_kv_mem, 1);

        // 5e: normed * norm_w * silu(z) -> cb_out
        cb_wait_front(cb_tmp2, D_TILES);
        cb_reserve_back(cb_out, D_TILES);
        init_sfpu(cb_tmp2, cb_out);
        for (uint32_t t = 0; t < D_TILES; t++) {
            tile_regs_acquire();
            copy_tile(cb_tmp2, t, 0);
            copy_tile(cb_norm_w, t, 1);
            mul_binary_tile_init();
            mul_binary_tile(0, 1, 0);
            copy_tile(cb_z, t, 1);
            copy_tile(cb_z, t, 2);
            sigmoid_tile_init();
            sigmoid_tile(1);
            mul_binary_tile_init();
            mul_binary_tile(1, 2, 1);
            mul_binary_tile(0, 1, 0);
            tile_regs_commit();
            tile_regs_wait();
            pack_tile(0, cb_out);
            tile_regs_release();
        }
        cb_push_back(cb_out, D_TILES);
        cb_pop_front(cb_tmp2, D_TILES);
        cb_pop_front(cb_z, D_TILES);
        cb_pop_front(cb_norm_w, D_TILES);

        // Scaler constants reused across batch iterations (not popped per iteration)
    }

    // Pop scaler constants ONCE after all iterations (reader pushed them once per head).
    // CBs must be empty at end of dispatch for program cache reuse.
    cb_pop_front(cb_scaler, 1);
    cb_pop_front(cb_eps, 1);
    cb_pop_front(cb_scaler_one, 1);
    cb_pop_front(cb_q_scale, 1);
}

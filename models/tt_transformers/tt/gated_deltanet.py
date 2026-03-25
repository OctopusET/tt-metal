# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Gated DeltaNet (linear attention) module for Qwen3.5.

All-device decode: projections, conv1d, L2 norm, gates, recurrence
(via fused kernel), and gated RMSNorm run on device. Zero host syncs.
State persists on device DRAM in fp32.

Reference: "Gated Delta Networks with Softmax Attention" (Yang et al., 2025)
"""

import math

import torch

import ttnn
from models.common.lightweightmodule import LightweightModule


def _mul_bcast(a, b, mem=None):
    """Multiply with COL broadcast support. Ensures smaller tensor is first (COL_A path)."""
    if a.shape[-1] > b.shape[-1]:
        a, b = b, a
    return ttnn.mul(a, b, fast_and_approximate_mode=True, memory_config=mem)


class GatedDeltaNet(LightweightModule):
    def __init__(self, mesh_device, args, state_dict, weight_cache_path, layer_num, dtype):
        super().__init__()
        self.args = args
        self.mesh_device = mesh_device
        self.layer_num = layer_num

        self.hidden_size = args.dim
        self.num_v_heads = args.linear_num_value_heads
        self.num_k_heads = args.linear_num_key_heads
        self.head_k_dim = args.linear_key_head_dim
        self.head_v_dim = args.linear_value_head_dim
        self.key_dim = self.head_k_dim * self.num_k_heads
        self.value_dim = self.head_v_dim * self.num_v_heads
        self.conv_dim = self.key_dim * 2 + self.value_dim
        self.conv_kernel_size = args.linear_conv_kernel_dim
        self.gqa_ratio = self.num_v_heads // self.num_k_heads
        self.scale = 1.0 / math.sqrt(self.head_k_dim)

        H = self.num_v_heads
        D_k = self.head_k_dim
        D_v = self.head_v_dim

        self.proj_compute_config = ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi2,
            math_approx_mode=False,
            fp32_dest_acc_en=False,
            packer_l1_acc=True,
        )

        layer_prefix = args.get_state_dict_prefix("GatedDeltaNet", layer_num)
        if args.dummy_weights or weight_cache_path is None:
            cache_name = lambda _: None
        else:
            cache_name = lambda name: weight_cache_path / f"{layer_prefix}.{name}"

        def load_weight(name, transpose=True):
            key = f"{layer_prefix}.{name}.weight"
            w = state_dict[key]
            if transpose and w.dim() == 2:
                w = w.transpose(-2, -1)
            return w

        def load_param(name):
            return state_dict[f"{layer_prefix}.{name}"]

        proj_dtype = dtype
        w_all = torch.cat(
            [load_weight("in_proj_qkv"), load_weight("in_proj_z"), load_weight("in_proj_b"), load_weight("in_proj_a")],
            dim=-1,
        )
        self._proj_splits = [self.conv_dim, self.value_dim, self.num_v_heads, self.num_v_heads]
        self._total_proj_dim = sum(self._proj_splits)
        self.in_proj_all = ttnn.as_tensor(
            w_all.unsqueeze(0).unsqueeze(0),
            dtype=proj_dtype,
            device=mesh_device,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            cache_file_name=cache_name("in_proj_all_fused"),
        )
        self.out_proj = ttnn.as_tensor(
            load_weight("out_proj").unsqueeze(0).unsqueeze(0),
            dtype=proj_dtype,
            device=mesh_device,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            cache_file_name=cache_name("out_proj"),
        )

        # Conv1d weights on device: 4 x [1, 1, 32, conv_dim]
        conv_weight_raw = state_dict[f"{layer_prefix}.conv1d.weight"].float().squeeze(1)  # (conv_dim, K)
        self._dev_conv_w = []
        for i in range(self.conv_kernel_size):
            w_pad = torch.zeros(1, 1, 32, self.conv_dim)
            w_pad[0, 0, 0, :] = conv_weight_raw[:, i]
            self._dev_conv_w.append(
                ttnn.as_tensor(
                    w_pad,
                    dtype=ttnn.bfloat16,
                    device=mesh_device,
                    layout=ttnn.TILE_LAYOUT,
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                    cache_file_name=cache_name(f"conv_w_{i}"),
                )
            )

        # Gate params on device: -A_exp [1,1,32,H], dt_bias [1,1,32,H]
        dt_bias = load_param("dt_bias").float()
        A_exp = load_param("A_log").float().exp()
        neg_A = torch.zeros(1, 1, 32, H)
        neg_A[0, 0, 0, :H] = -A_exp
        self._dev_neg_A = ttnn.from_torch(
            neg_A,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=mesh_device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        db = torch.zeros(1, 1, 32, H)
        db[0, 0, 0, :H] = dt_bias
        self._dev_dt_bias = ttnn.from_torch(
            db,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=mesh_device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

        # Norm weight [1,1,1,D_v] shared across heads
        norm_w = state_dict[f"{layer_prefix}.norm.weight"].float()
        self._dev_norm_w = ttnn.from_torch(
            norm_w.reshape(1, 1, 1, D_v),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=mesh_device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

        # Pre-allocate ones for fused kernel beta=1
        self._dev_ones = ttnn.from_torch(
            torch.ones(1, H, 1, 1),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=mesh_device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

        # Host fallback (kept for potential debugging)
        self._conv_w_host = conv_weight_raw.T
        self._dt_bias_host = dt_bias
        self._A_exp_host = A_exp
        self._norm_w_host = norm_w

    @property
    def layer_past(self):
        return None

    @layer_past.setter
    def layer_past(self, value):
        pass

    def initialize_states(self, batch_size=1, B_pad=32):
        H = self.num_v_heads
        D_k = self.head_k_dim
        D_v = self.head_v_dim

        # Device state (fp32, persistent across tokens)
        self._dev_state = ttnn.from_torch(
            torch.zeros(1, H, D_k, D_v),
            dtype=ttnn.float32,
            layout=ttnn.TILE_LAYOUT,
            device=self.mesh_device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

        # Conv1d: 4 device buffers for ring buffer
        self._conv_bufs = []
        for _ in range(self.conv_kernel_size):
            self._conv_bufs.append(
                ttnn.from_torch(
                    torch.zeros(1, 1, B_pad, self.conv_dim),
                    dtype=ttnn.bfloat16,
                    layout=ttnn.TILE_LAYOUT,
                    device=self.mesh_device,
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                )
            )
        self._conv_pos = 0

        # Host fallback state
        self._host_state = torch.zeros(H, D_k, D_v)
        self._conv_state = torch.zeros(self.conv_kernel_size, self.conv_dim)
        self._out_pad = torch.zeros(1, 1, B_pad, self.value_dim)

    def forward(self, x):
        B_pad = x.shape[2]
        H = self.num_v_heads
        K_h = self.num_k_heads
        D_k = self.head_k_dim
        D_v = self.head_v_dim
        L1 = ttnn.L1_MEMORY_CONFIG
        DRAM = ttnn.DRAM_MEMORY_CONFIG

        # 1. Projection

        _p = lambda s: print(s, end="", flush=True)
        _p("1.")
        all_proj = ttnn.linear(x, self.in_proj_all, compute_kernel_config=self.proj_compute_config)

        # 2. Split into QKV, Z, B, A
        _p("2.")
        s = self._proj_splits
        qkv = ttnn.slice(all_proj, [0, 0, 0, 0], [1, 1, B_pad, s[0]], memory_config=L1)
        z = ttnn.slice(all_proj, [0, 0, 0, s[0]], [1, 1, B_pad, s[0] + s[1]], memory_config=L1)
        b_proj = ttnn.slice(all_proj, [0, 0, 0, s[0] + s[1]], [1, 1, B_pad, s[0] + s[1] + s[2]], memory_config=L1)
        a_proj = ttnn.slice(
            all_proj, [0, 0, 0, s[0] + s[1] + s[2]], [1, 1, B_pad, self._total_proj_dim], memory_config=L1
        )
        ttnn.deallocate(all_proj)

        # 3. Conv1d (4-buffer circular on device)
        _p("3.")
        ttnn.deallocate(self._conv_bufs[self._conv_pos])
        self._conv_bufs[self._conv_pos] = qkv
        conv_out = None
        for i in range(self.conv_kernel_size):
            buf_idx = (self._conv_pos + 1 + i) % self.conv_kernel_size
            weighted = _mul_bcast(self._conv_bufs[buf_idx], self._dev_conv_w[i], L1)
            if conv_out is None:
                conv_out = weighted
            else:
                conv_out = ttnn.add(conv_out, weighted, memory_config=L1)
                ttnn.deallocate(weighted)
        self._conv_pos = (self._conv_pos + 1) % self.conv_kernel_size
        conv_out = ttnn.silu(conv_out, memory_config=L1)

        # 4. Split Q, K, V
        _p("4.")
        q_flat = ttnn.slice(conv_out, [0, 0, 0, 0], [1, 1, B_pad, self.key_dim], memory_config=L1)
        k_flat = ttnn.slice(conv_out, [0, 0, 0, self.key_dim], [1, 1, B_pad, 2 * self.key_dim], memory_config=L1)
        v_flat = ttnn.slice(conv_out, [0, 0, 0, 2 * self.key_dim], [1, 1, B_pad, self.conv_dim], memory_config=L1)
        ttnn.deallocate(conv_out)

        # 5. Reshape to heads
        _p("5.")
        q_heads = ttnn.reshape(q_flat, [1, K_h, B_pad, D_k])
        k_heads = ttnn.reshape(k_flat, [1, K_h, B_pad, D_k])
        v_heads = ttnn.reshape(v_flat, [1, H, B_pad, D_v])
        ttnn.deallocate(q_flat)
        ttnn.deallocate(k_flat)
        ttnn.deallocate(v_flat)

        # 6. GQA expand
        _p("6.")
        if self.gqa_ratio > 1:
            q_heads = ttnn.repeat_interleave(q_heads, self.gqa_ratio, dim=1)
            k_heads = ttnn.repeat_interleave(k_heads, self.gqa_ratio, dim=1)

        # 7. L2 normalize Q and K
        _p("7.")
        q_sq = _mul_bcast(q_heads, q_heads, L1)
        q_norm_sq = ttnn.sum(q_sq, dim=3, keepdim=True, memory_config=L1)
        ttnn.deallocate(q_sq)
        q_inv = ttnn.rsqrt(q_norm_sq, memory_config=L1)
        ttnn.deallocate(q_norm_sq)
        q_heads = _mul_bcast(q_heads, q_inv, L1)  # COL_A broadcast
        ttnn.deallocate(q_inv)
        q_heads = ttnn.multiply(q_heads, self.scale, memory_config=L1)

        k_sq = _mul_bcast(k_heads, k_heads, L1)
        k_norm_sq = ttnn.sum(k_sq, dim=3, keepdim=True, memory_config=L1)
        ttnn.deallocate(k_sq)
        k_inv = ttnn.rsqrt(k_norm_sq, memory_config=L1)
        ttnn.deallocate(k_norm_sq)
        k_heads = _mul_bcast(k_heads, k_inv, L1)  # COL_A broadcast
        ttnn.deallocate(k_inv)

        # 8. Slice to single token [1, H, 1, D]
        _p("8.")
        q = ttnn.slice(q_heads, [0, 0, 0, 0], [1, H, 1, D_k], memory_config=L1)
        k_row = ttnn.slice(k_heads, [0, 0, 0, 0], [1, H, 1, D_k], memory_config=L1)
        v = ttnn.slice(v_heads, [0, 0, 0, 0], [1, H, 1, D_v], memory_config=L1)
        ttnn.deallocate(q_heads)
        ttnn.deallocate(k_heads)
        ttnn.deallocate(v_heads)

        # 9. Gates
        _p("9.")
        b_val = ttnn.slice(b_proj, [0, 0, 0, 0], [1, 1, 1, H], memory_config=L1)
        a_val = ttnn.slice(a_proj, [0, 0, 0, 0], [1, 1, 1, H], memory_config=L1)
        ttnn.deallocate(b_proj)
        ttnn.deallocate(a_proj)

        beta = ttnn.sigmoid(b_val, memory_config=L1)
        ttnn.deallocate(b_val)

        dt_bias_1 = ttnn.slice(self._dev_dt_bias, [0, 0, 0, 0], [1, 1, 1, H], memory_config=L1)
        neg_A_1 = ttnn.slice(self._dev_neg_A, [0, 0, 0, 0], [1, 1, 1, H], memory_config=L1)
        a_biased = ttnn.add(a_val, dt_bias_1, memory_config=L1)
        ttnn.deallocate(a_val)
        ttnn.deallocate(dt_bias_1)
        a_sp = ttnn.softplus(a_biased, memory_config=L1)
        ttnn.deallocate(a_biased)
        decay_flat = _mul_bcast(neg_A_1, a_sp, L1)
        ttnn.deallocate(neg_A_1)
        ttnn.deallocate(a_sp)
        decay_flat = ttnn.exp(decay_flat, memory_config=L1)

        decay = ttnn.reshape(decay_flat, [1, H, 1, 1])
        beta_4d = ttnn.reshape(beta, [1, H, 1, 1])
        ttnn.deallocate(decay_flat)
        ttnn.deallocate(beta)

        # 10. Retrieve from state: kv_mem = k_row @ state
        _p("10.")
        kv_mem = ttnn.matmul(k_row, self._dev_state, memory_config=L1)

        # 11. Delta = (v - kv_mem) * beta
        _p("11.")
        delta = ttnn.sub(v, kv_mem, memory_config=L1)
        ttnn.deallocate(kv_mem)
        ttnn.deallocate(v)
        delta = _mul_bcast(delta, beta_4d, L1)  # COL_A broadcast [1,H,1,D] * [1,H,1,1]
        ttnn.deallocate(beta_4d)

        # 12. k as column vector [1,H,D,1]
        _p("12.")
        k_col = ttnn.transpose(k_row, 2, 3)
        ttnn.deallocate(k_row)

        # 13. Fused kernel: decay + outer product + output
        _p("13a.")
        q = ttnn.to_memory_config(q, DRAM)
        _p("13b.")
        k_col = ttnn.to_memory_config(k_col, DRAM)
        _p("13c.")
        delta = ttnn.to_memory_config(delta, DRAM)
        _p("13d.")
        decay = ttnn.to_memory_config(decay, DRAM)
        _p(f"13e(q={q.memory_config()},k={k_col.memory_config()},d={delta.memory_config()}).")
        output_heads, new_state = ttnn.experimental.gated_delta_net(
            q, k_col, delta, decay, self._dev_ones, self._dev_state
        )
        ttnn.deallocate(q)
        ttnn.deallocate(k_col)
        ttnn.deallocate(delta)
        ttnn.deallocate(decay)
        self._dev_state = new_state
        _p("13ok.")

        # 14. Gated RMSNorm: normed = x * rsqrt(mean(x^2)) * weight * silu(z)
        _p("14sync.")
        ttnn.synchronize_device(self.mesh_device)
        _p("14.")
        out_sq = _mul_bcast(output_heads, output_heads, L1)
        variance = ttnn.mean(out_sq, dim=3, keepdim=True, memory_config=L1)  # [1,H,1,1]
        ttnn.deallocate(out_sq)
        inv_rms = ttnn.rsqrt(variance, memory_config=L1)
        ttnn.deallocate(variance)
        output_normed = _mul_bcast(output_heads, inv_rms, L1)  # COL_A broadcast
        ttnn.deallocate(output_heads)
        ttnn.deallocate(inv_rms)
        output_normed = _mul_bcast(output_normed, self._dev_norm_w, L1)  # [1,H,1,D] * [1,1,1,D] = broadcast on dim 1

        # Z gate
        z_token = ttnn.slice(z, [0, 0, 0, 0], [1, 1, 1, self.value_dim], memory_config=L1)
        ttnn.deallocate(z)
        z_heads = ttnn.reshape(z_token, [1, H, 1, D_v])
        ttnn.deallocate(z_token)
        z_gate = ttnn.silu(z_heads, memory_config=L1)
        ttnn.deallocate(z_heads)
        output_gated = _mul_bcast(output_normed, z_gate, L1)
        ttnn.deallocate(output_normed)
        ttnn.deallocate(z_gate)

        # 15. Output via host pad (ttnn.pad output incompatible with sharded residual config)
        _p("15.")
        output_cpu = ttnn.to_torch(output_gated).float().reshape(1, self.value_dim)
        ttnn.deallocate(output_gated)
        self._out_pad.zero_()
        self._out_pad[0, 0, 0, :] = output_cpu
        output_flat = ttnn.from_torch(
            self._out_pad,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=self.mesh_device,
            memory_config=DRAM,
        )

        # 16. Output projection
        _p("16.")
        output = ttnn.linear(
            output_flat, self.out_proj, compute_kernel_config=self.proj_compute_config, memory_config=DRAM
        )
        _p("ok\n")
        return output

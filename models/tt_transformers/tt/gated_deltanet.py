# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Gated DeltaNet (linear attention) module for Qwen3.5.

Runs fully on TT device -- zero host syncs per layer.
All operations use ttnn ops with bf16 tensors.
State is persistent on device DRAM.
"""

import math

import torch

import ttnn
from models.common.lightweightmodule import LightweightModule


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

        # Projection weights (device DRAM)
        proj_dtype = dtype
        w_all = torch.cat(
            [load_weight("in_proj_qkv"), load_weight("in_proj_z"), load_weight("in_proj_b"), load_weight("in_proj_a")],
            dim=-1,
        )
        self._proj_splits = [self.conv_dim, self.value_dim, self.num_v_heads, self.num_v_heads]
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

        # Conv1d: stack all kernel weights as [1, 1, K, conv_dim] for single multiply+sum
        conv_weight_raw = state_dict[f"{layer_prefix}.conv1d.weight"].float().squeeze(1)  # [conv_dim, K]
        # [K, conv_dim] -> [1, 1, K, conv_dim]
        self._conv_w_stacked = ttnn.from_torch(
            conv_weight_raw.T.unsqueeze(0).unsqueeze(0).contiguous(),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=mesh_device,
        )

        # Gate params: pre-compute -A_exp on device
        dt_bias = load_param("dt_bias").float()
        A_exp = load_param("A_log").float().exp()
        self._dt_bias = ttnn.from_torch(
            dt_bias.reshape(1, self.num_v_heads, 1, 1), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh_device
        )
        self._neg_A_exp = ttnn.from_torch(
            (-A_exp).reshape(1, self.num_v_heads, 1, 1),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=mesh_device,
        )

        # Norm weight: [1, 1, 1, head_v_dim] (shared across heads, broadcast)
        norm_w = state_dict[f"{layer_prefix}.norm.weight"].float()
        self._norm_w = ttnn.from_torch(
            norm_w.reshape(1, 1, 1, self.head_v_dim), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh_device
        )

    @property
    def layer_past(self):
        return None

    @layer_past.setter
    def layer_past(self, value):
        pass

    def initialize_states(self, batch_size=1):
        """Initialize device-side recurrent state and conv ring buffer."""
        self._device_state = ttnn.from_torch(
            torch.zeros(1, self.num_v_heads, self.head_k_dim, self.head_v_dim),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=self.mesh_device,
        )
        # Conv state as single stacked tensor: [1, 1, K, conv_dim]
        self._conv_state = ttnn.from_torch(
            torch.zeros(1, 1, self.conv_kernel_size, self.conv_dim),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=self.mesh_device,
        )

    def forward(self, x):
        """Single-token decode forward -- fully on device, zero host syncs."""
        B_pad = x.shape[2]
        x_tok = x[:, :, :1, :]

        # 1. Fused projection: [1,1,1,hidden] -> [1,1,1,total_proj]
        all_proj = ttnn.linear(x_tok, self.in_proj_all, compute_kernel_config=self.proj_compute_config)

        # 2. Split: QKV | Z | B | A (1 op)
        s = self._proj_splits
        qkv_raw, z_raw, b_raw, a_raw = ttnn.split(all_proj, [s[0], s[1], s[2], s[3]], dim=3)
        ttnn.deallocate(all_proj)

        # 3. Conv1d ring buffer: shift + weighted sum (3 ops instead of 8)
        # Shift state: drop oldest row, append new. [1,1,K,conv_dim]
        # qkv_raw is [1,1,1,conv_dim], need to append as last row
        new_rows = ttnn.concat([self._conv_state[:, :, 1:, :], qkv_raw], dim=2)
        ttnn.deallocate(self._conv_state)
        self._conv_state = new_rows

        # Weighted sum: element-wise multiply then sum over K dimension
        conv_prod = ttnn.multiply(self._conv_state, self._conv_w_stacked)  # [1,1,K,conv_dim]
        conv_out = ttnn.sum(conv_prod, dim=2, keepdim=True)  # [1,1,1,conv_dim]
        ttnn.deallocate(conv_prod)
        conv_out = ttnn.silu(conv_out)

        # 4. Split Q, K, V + reshape to multi-head (4 ops)
        q_flat, k_flat, v_flat = ttnn.split(conv_out, [self.key_dim, self.key_dim, self.value_dim], dim=3)
        ttnn.deallocate(conv_out)
        q = ttnn.reshape(q_flat, (1, self.num_k_heads, 1, self.head_k_dim))
        k = ttnn.reshape(k_flat, (1, self.num_k_heads, 1, self.head_k_dim))
        v = ttnn.reshape(v_flat, (1, self.num_v_heads, 1, self.head_v_dim))

        # GQA expand (2 ops)
        if self.gqa_ratio > 1:
            q = ttnn.repeat_interleave(q, self.gqa_ratio, dim=1)
            k = ttnn.repeat_interleave(k, self.gqa_ratio, dim=1)

        # 5. L2 normalize Q and K (6 ops total for both)
        q_sq_sum = ttnn.sum(ttnn.multiply(q, q), dim=-1, keepdim=True)
        q = ttnn.multiply(ttnn.multiply(q, ttnn.rsqrt(q_sq_sum)), self.scale)
        k_sq_sum = ttnn.sum(ttnn.multiply(k, k), dim=-1, keepdim=True)
        k = ttnn.multiply(k, ttnn.rsqrt(k_sq_sum))

        # 6. Gates (5 ops instead of 7)
        b = ttnn.reshape(b_raw, (1, self.num_v_heads, 1, 1))
        a = ttnn.reshape(a_raw, (1, self.num_v_heads, 1, 1))
        beta = ttnn.sigmoid(b)
        # decay = exp(-A * softplus(a + dt_bias)) = exp(neg_A * log1p(exp(a + dt_bias)))
        decay = ttnn.exp(ttnn.multiply(self._neg_A_exp, ttnn.log1p(ttnn.exp(ttnn.add(a, self._dt_bias)))))

        # 7. Recurrence (8 ops)
        self._device_state = ttnn.multiply(self._device_state, decay)
        kv_mem = ttnn.matmul(k, self._device_state)
        delta = ttnn.multiply(ttnn.subtract(v, kv_mem), beta)
        k_t = ttnn.permute(k, (0, 1, 3, 2))
        self._device_state = ttnn.add(self._device_state, ttnn.matmul(k_t, delta))
        output = ttnn.matmul(q, self._device_state)

        # 8. Gated RMSNorm (5 ops)
        z = ttnn.reshape(z_raw, (1, self.num_v_heads, 1, self.head_v_dim))
        variance = ttnn.mean(ttnn.multiply(output, output), dim=-1, keepdim=True)
        output_normed = ttnn.multiply(ttnn.multiply(output, ttnn.rsqrt(variance)), self._norm_w)
        output_gated = ttnn.multiply(output_normed, ttnn.silu(z))

        # 9. Merge heads + pad + output projection (3 ops)
        output_flat = ttnn.reshape(output_gated, (1, 1, 1, self.value_dim))
        if B_pad > 1:
            output_flat = ttnn.pad(output_flat, [1, 1, B_pad, self.value_dim], [0, 0, 0, 0], 0.0)
        return ttnn.linear(output_flat, self.out_proj, compute_kernel_config=self.proj_compute_config)

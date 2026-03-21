# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Gated DeltaNet (linear attention) module for Qwen3.5.

Runs fully on TT device -- zero host syncs per layer.
All operations (projections, conv1d, recurrence, norms) use ttnn ops
with bf16 tensors. State is persistent on device DRAM.

Reference: "Gated Delta Networks with Softmax Attention" (Yang et al., 2025)
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

        # Architecture params
        self.hidden_size = args.dim
        self.num_v_heads = args.linear_num_value_heads
        self.num_k_heads = args.linear_num_key_heads
        self.head_k_dim = args.linear_key_head_dim  # 128
        self.head_v_dim = args.linear_value_head_dim  # 128
        self.key_dim = self.head_k_dim * self.num_k_heads
        self.value_dim = self.head_v_dim * self.num_v_heads
        self.conv_dim = self.key_dim * 2 + self.value_dim
        self.conv_kernel_size = args.linear_conv_kernel_dim  # 4
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

        # Conv1d weights on device: [conv_kernel_size, 1, 1, conv_dim]
        conv_weight_raw = state_dict[f"{layer_prefix}.conv1d.weight"].float().squeeze(1)  # [conv_dim, K]
        # Store each kernel position as a device tensor for element-wise multiply
        self._conv_weights = []
        for i in range(self.conv_kernel_size):
            cw = conv_weight_raw[:, i].unsqueeze(0).unsqueeze(0).unsqueeze(0)  # [1,1,1,conv_dim]
            self._conv_weights.append(
                ttnn.from_torch(cw, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh_device)
            )

        # Gate params on device
        dt_bias = load_param("dt_bias").float()
        A_exp = load_param("A_log").float().exp()
        self._dt_bias = ttnn.from_torch(
            dt_bias.reshape(1, self.num_v_heads, 1, 1), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh_device
        )
        self._A_exp = ttnn.from_torch(
            A_exp.reshape(1, self.num_v_heads, 1, 1), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh_device
        )

        # Norm weight on device: [1, 1, 1, head_v_dim] (shared across heads, broadcast)
        norm_w = state_dict[f"{layer_prefix}.norm.weight"].float()
        self._norm_w = ttnn.from_torch(
            norm_w.reshape(1, 1, 1, self.head_v_dim),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=mesh_device,
        )

    @property
    def layer_past(self):
        return None

    @layer_past.setter
    def layer_past(self, value):
        pass

    def initialize_states(self, batch_size=1):
        """Initialize device-side recurrent state and conv buffer."""
        # State: [1, num_v_heads, head_k_dim, head_v_dim] on device
        self._device_state = ttnn.from_torch(
            torch.zeros(1, self.num_v_heads, self.head_k_dim, self.head_v_dim),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=self.mesh_device,
        )
        # Conv state: list of conv_kernel_size tensors, each [1,1,1,conv_dim]
        self._conv_state = []
        for _ in range(self.conv_kernel_size):
            self._conv_state.append(
                ttnn.from_torch(
                    torch.zeros(1, 1, 1, self.conv_dim),
                    dtype=ttnn.bfloat16,
                    layout=ttnn.TILE_LAYOUT,
                    device=self.mesh_device,
                )
            )

    def forward(self, x):
        """Single-token decode forward -- fully on device, zero host syncs.

        Args:
            x: (1, 1, B_pad, hidden_size) on device
        Returns:
            output: (1, 1, B_pad, hidden_size) on device
        """
        # Slice to single token: [1, 1, 1, hidden_size]
        x_tok = x[:, :, :1, :]

        # 1. Fused projection
        all_proj = ttnn.linear(x_tok, self.in_proj_all, compute_kernel_config=self.proj_compute_config)

        # 2. Split: QKV | Z | B | A
        s = self._proj_splits
        # Split into [conv_dim, value_dim, num_v_heads, num_v_heads]
        qkv_raw, z_raw, b_raw, a_raw = ttnn.split(all_proj, [s[0], s[1], s[2], s[3]], dim=3)
        ttnn.deallocate(all_proj)

        # 3. Conv1d: shift state, add new, weighted sum + silu
        # Shift conv state (ring buffer)
        old_state_0 = self._conv_state[0]
        for i in range(self.conv_kernel_size - 1):
            self._conv_state[i] = self._conv_state[i + 1]
        self._conv_state[self.conv_kernel_size - 1] = qkv_raw
        ttnn.deallocate(old_state_0)

        # Weighted sum: sum(conv_state[i] * conv_weight[i])
        conv_out = ttnn.multiply(self._conv_state[0], self._conv_weights[0])
        for i in range(1, self.conv_kernel_size):
            term = ttnn.multiply(self._conv_state[i], self._conv_weights[i])
            conv_out = ttnn.add(conv_out, term)
            ttnn.deallocate(term)
        conv_out = ttnn.silu(conv_out)  # [1, 1, 1, conv_dim]

        # 4. Split Q, K, V from conv output
        q_flat, k_flat, v_flat = ttnn.split(conv_out, [self.key_dim, self.key_dim, self.value_dim], dim=3)
        ttnn.deallocate(conv_out)

        # Reshape to multi-head: [1, H, 1, D]
        q = ttnn.reshape(q_flat, (1, self.num_k_heads, 1, self.head_k_dim))
        k = ttnn.reshape(k_flat, (1, self.num_k_heads, 1, self.head_k_dim))
        v = ttnn.reshape(v_flat, (1, self.num_v_heads, 1, self.head_v_dim))
        ttnn.deallocate(q_flat)
        ttnn.deallocate(k_flat)
        ttnn.deallocate(v_flat)

        # GQA expand: [1, num_k_heads, 1, D] -> [1, num_v_heads, 1, D]
        if self.gqa_ratio > 1:
            q = ttnn.repeat_interleave(q, self.gqa_ratio, dim=1)
            k = ttnn.repeat_interleave(k, self.gqa_ratio, dim=1)

        # 5. L2 normalize Q and K
        def l2_normalize(t):
            sq = ttnn.multiply(t, t)
            sq_sum = ttnn.sum(sq, dim=-1, keepdim=True)
            ttnn.deallocate(sq)
            inv_norm = ttnn.rsqrt(sq_sum)
            ttnn.deallocate(sq_sum)
            return ttnn.multiply(t, inv_norm)

        q = ttnn.multiply(l2_normalize(q), self.scale)
        k = l2_normalize(k)

        # 6. Gates: beta = sigmoid(b), decay = exp(-A * softplus(a + dt_bias))
        # b_raw, a_raw: [1, 1, 1, num_v_heads] -> reshape to [1, H, 1, 1]
        b = ttnn.reshape(b_raw, (1, self.num_v_heads, 1, 1))
        a = ttnn.reshape(a_raw, (1, self.num_v_heads, 1, 1))
        ttnn.deallocate(b_raw)
        ttnn.deallocate(a_raw)

        beta = ttnn.sigmoid(b)
        ttnn.deallocate(b)
        # softplus(a + dt_bias) = log(1 + exp(a + dt_bias))
        a_biased = ttnn.add(a, self._dt_bias)
        ttnn.deallocate(a)
        softplus_a = ttnn.log1p(ttnn.exp(a_biased))
        ttnn.deallocate(a_biased)
        decay = ttnn.exp(ttnn.neg(ttnn.multiply(self._A_exp, softplus_a)))
        ttnn.deallocate(softplus_a)

        # 7. Recurrence (all on device, bf16)
        # state [1, H, D, D], q/k [1, H, 1, D], v [1, H, 1, D]
        # decay [1, H, 1, 1], beta [1, H, 1, 1]

        # state *= decay (broadcast)
        self._device_state = ttnn.multiply(self._device_state, decay)
        ttnn.deallocate(decay)

        # kv_mem = k @ state -> [1, H, 1, D]
        kv_mem = ttnn.matmul(k, self._device_state)

        # delta = (v - kv_mem) * beta
        delta = ttnn.subtract(v, kv_mem)
        ttnn.deallocate(kv_mem)
        delta = ttnn.multiply(delta, beta)
        ttnn.deallocate(beta)

        # state += k^T @ delta (rank-1 update)
        k_t = ttnn.permute(k, (0, 1, 3, 2))  # [1, H, D, 1]
        update = ttnn.matmul(k_t, delta)  # [1, H, D, D]
        ttnn.deallocate(k_t)
        ttnn.deallocate(delta)
        self._device_state = ttnn.add(self._device_state, update)
        ttnn.deallocate(update)

        # output = q @ state -> [1, H, 1, D]
        output = ttnn.matmul(q, self._device_state)

        # 8. Gated RMSNorm + head merge
        # z_raw: [1, 1, 1, value_dim] -> [1, H, 1, D]
        z = ttnn.reshape(z_raw, (1, self.num_v_heads, 1, self.head_v_dim))
        ttnn.deallocate(z_raw)

        # RMSNorm: output / sqrt(mean(output^2) + eps)
        out_sq = ttnn.multiply(output, output)
        variance = ttnn.mean(out_sq, dim=-1, keepdim=True)
        ttnn.deallocate(out_sq)
        eps_tensor = ttnn.full_like(variance, self.args.norm_eps)
        inv_rms = ttnn.rsqrt(ttnn.add(variance, eps_tensor))
        ttnn.deallocate(variance)
        ttnn.deallocate(eps_tensor)
        output_normed = ttnn.multiply(output, inv_rms)
        ttnn.deallocate(inv_rms)
        output_normed = ttnn.multiply(output_normed, self._norm_w)

        # Gate: output * silu(z)
        output_gated = ttnn.multiply(output_normed, ttnn.silu(z))
        ttnn.deallocate(output_normed)

        # Merge heads: [1, H, 1, D] -> [1, 1, 1, H*D]
        output_flat = ttnn.reshape(output_gated, (1, 1, 1, self.value_dim))
        ttnn.deallocate(output_gated)

        # Pad to batch: [1, 1, 1, value_dim] -> [1, 1, B_pad, value_dim]
        B_pad = x.shape[2]
        if B_pad > 1:
            output_flat = ttnn.pad(output_flat, [1, 1, B_pad, self.value_dim], [0, 0, 0, 0], 0.0)

        # 9. Output projection
        output_final = ttnn.linear(output_flat, self.out_proj, compute_kernel_config=self.proj_compute_config)
        ttnn.deallocate(output_flat)

        return output_final

# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Gated DeltaNet (linear attention) module for Qwen3.5.

Hybrid device/host approach:
- Projection matmuls: on device (expensive, bandwidth-bound)
- Conv1d, L2 norm, gates, gated RMSNorm: on host (cheap, tiny tensors)
- Recurrence: fused device kernel (state stays on device DRAM, 1 kernel launch)
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

        # Conv, gate, norm weights on HOST
        conv_weight_raw = state_dict[f"{layer_prefix}.conv1d.weight"].float().squeeze(1)
        self._conv_w = conv_weight_raw.T
        self._dt_bias = load_param("dt_bias").float()
        self._A_exp = load_param("A_log").float().exp()
        self._norm_w = state_dict[f"{layer_prefix}.norm.weight"].float()

    @property
    def layer_past(self):
        return None

    @layer_past.setter
    def layer_past(self, value):
        pass

    def initialize_states(self, batch_size=1):
        """Initialize device-side recurrent state and host-side conv buffer."""
        # State on DEVICE (fp32 for precision) -- stays on device between tokens
        self._device_state = ttnn.from_torch(
            torch.zeros(1, self.num_v_heads, self.head_k_dim, self.head_v_dim),
            dtype=ttnn.float32,
            layout=ttnn.TILE_LAYOUT,
            device=self.mesh_device,
        )
        # Conv state on HOST (tiny, fast)
        self._conv_state = torch.zeros(self.conv_kernel_size, self.conv_dim)

    def forward(self, x):
        """Single-token decode: projections on device, recurrence via fused kernel.

        State stays on device DRAM between tokens. No state transfer via host.
        """
        B_pad = x.shape[2]

        # 1. Device: fused projection matmul
        all_proj = ttnn.linear(x, self.in_proj_all, compute_kernel_config=self.proj_compute_config)

        # 2. Sync: transfer projection output to host
        F = torch.nn.functional
        all_h = ttnn.to_torch(all_proj).float()[0, 0, 0, :]
        ttnn.deallocate(all_proj)
        s = self._proj_splits
        qkv_h = all_h[: s[0]]
        z_h = all_h[s[0] : s[0] + s[1]]
        b_h = all_h[s[0] + s[1] : s[0] + s[1] + s[2]]
        a_h = all_h[s[0] + s[1] + s[2] :]

        # 3. Host: conv1d ring buffer
        self._conv_state[:-1] = self._conv_state[1:].clone()
        self._conv_state[-1] = qkv_h
        qkv_h = F.silu((self._conv_state * self._conv_w).sum(dim=0))

        # 4. Host: split Q, K, V + GQA expand + L2 normalize
        q_h = qkv_h[: self.key_dim].reshape(self.num_k_heads, self.head_k_dim)
        k_h = qkv_h[self.key_dim : 2 * self.key_dim].reshape(self.num_k_heads, self.head_k_dim)
        v_h = qkv_h[2 * self.key_dim :].reshape(self.num_v_heads, self.head_v_dim)
        if self.gqa_ratio > 1:
            q_h = q_h.repeat_interleave(self.gqa_ratio, dim=0)
            k_h = k_h.repeat_interleave(self.gqa_ratio, dim=0)
        q_h = F.normalize(q_h, dim=-1) * self.scale
        k_h = F.normalize(k_h, dim=-1)

        # 5. Host: gates
        beta = b_h.sigmoid()
        decay = (-self._A_exp * F.softplus(a_h + self._dt_bias)).exp()

        # 6. Send Q, K, V, decay, beta to device for fused kernel
        q_tt = ttnn.from_torch(
            q_h.reshape(1, self.num_v_heads, 1, self.head_k_dim),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=self.mesh_device,
        )
        k_tt = ttnn.from_torch(
            k_h.reshape(1, self.num_v_heads, 1, self.head_k_dim),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=self.mesh_device,
        )
        v_tt = ttnn.from_torch(
            v_h.reshape(1, self.num_v_heads, 1, self.head_v_dim),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=self.mesh_device,
        )
        decay_tt = ttnn.from_torch(
            decay.reshape(1, self.num_v_heads, 1, 1),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=self.mesh_device,
        )
        beta_tt = ttnn.from_torch(
            beta.reshape(1, self.num_v_heads, 1, 1),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=self.mesh_device,
        )

        # 7. Device: fused recurrence kernel (1 kernel launch)
        # Convert fp32 state to bf16 for kernel, convert result back to fp32
        state_bf16 = ttnn.typecast(self._device_state, ttnn.bfloat16)
        output_tt, new_state_bf16 = ttnn.experimental.gated_delta_net(q_tt, k_tt, v_tt, decay_tt, beta_tt, state_bf16)
        self._device_state = ttnn.typecast(new_state_bf16, ttnn.float32)

        # 8. Sync: get recurrence output for gated RMSNorm on host
        output_h = ttnn.to_torch(output_tt).float()[0, :, 0, : self.head_v_dim]  # [H, D]
        ttnn.deallocate(output_tt)

        # 9. Host: gated RMSNorm + head merge
        z_heads = z_h.reshape(self.num_v_heads, self.head_v_dim)
        variance = output_h.pow(2).mean(-1, keepdim=True)
        output_normed = output_h * torch.rsqrt(variance + self.args.norm_eps)
        output_normed = output_normed * self._norm_w
        output_gated = output_normed * F.silu(z_heads)
        output_flat = output_gated.reshape(1, self.value_dim)

        # 10. Device: output projection
        out_pad = torch.zeros(1, 1, B_pad, self.value_dim)
        out_pad[0, 0, 0, :] = output_flat
        output = ttnn.from_torch(
            out_pad,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=self.mesh_device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        output = ttnn.linear(output, self.out_proj, compute_kernel_config=self.proj_compute_config)

        return output

# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Gated DeltaNet (linear attention) module for Qwen3.5.

Fully device-side: projections, conv1d, fused kernel. Zero host sync.

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

        # Conv weights on device: 4 rows
        conv_weight_raw = state_dict[f"{layer_prefix}.conv1d.weight"].float().squeeze(1)
        conv_w_host = conv_weight_raw.T
        B = getattr(args, "tile_padded_batch_rows", 32)
        self._conv_w_devs = []
        for r in range(self.conv_kernel_size):
            w_pad = torch.zeros(1, 1, B, self.conv_dim, dtype=torch.bfloat16)
            w_pad[0, 0, 0, :] = conv_w_host[r].bfloat16()
            self._conv_w_devs.append(
                ttnn.from_torch(
                    w_pad, layout=ttnn.TILE_LAYOUT, device=mesh_device, memory_config=ttnn.DRAM_MEMORY_CONFIG
                )
            )

        # Constant device tensors
        dt_bias = load_param("dt_bias").float()
        A_exp = load_param("A_log").float().exp()
        norm_w = state_dict[f"{layer_prefix}.norm.weight"].float()

        self._dt_bias_dev = ttnn.from_torch(
            dt_bias.reshape(1, self.num_v_heads, 1, 1).bfloat16(),
            layout=ttnn.TILE_LAYOUT,
            device=mesh_device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        self._neg_A_exp_dev = ttnn.from_torch(
            (-A_exp).reshape(1, self.num_v_heads, 1, 1).bfloat16(),
            layout=ttnn.TILE_LAYOUT,
            device=mesh_device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        self._norm_w_dev = ttnn.from_torch(
            norm_w.unsqueeze(0).expand(self.num_v_heads, -1).unsqueeze(0).unsqueeze(2).contiguous().bfloat16(),
            layout=ttnn.TILE_LAYOUT,
            device=mesh_device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

    @property
    def layer_past(self):
        return None

    @layer_past.setter
    def layer_past(self, value):
        pass

    def initialize_states(self, batch_size=1, B_pad=32):
        H, D = self.num_v_heads, self.head_v_dim
        # Conv state: 4 device tensors (circular buffer)
        self._conv_rows = []
        for _ in range(self.conv_kernel_size):
            self._conv_rows.append(
                ttnn.from_torch(
                    torch.zeros(1, 1, B_pad, self.conv_dim, dtype=torch.bfloat16),
                    layout=ttnn.TILE_LAYOUT,
                    device=self.mesh_device,
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                )
            )
        self._oldest = 0
        # Recurrent state on device: [batch_size, H, K, D]
        self._dev_state = ttnn.from_torch(
            torch.zeros(batch_size, H, self.head_k_dim, D),
            dtype=ttnn.float32,
            layout=ttnn.TILE_LAYOUT,
            device=self.mesh_device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

    def forward(self, x):
        B = x.shape[2]
        s = self._proj_splits

        # 1. Projection on device
        proj = ttnn.linear(x, self.in_proj_all, compute_kernel_config=self.proj_compute_config)

        # 2. Conv state update: in-place copy (preserves tensor addresses for trace)
        qkv_new = ttnn.slice(proj, [0, 0, 0, 0], [1, 1, B, self.conv_dim])
        ttnn.copy(qkv_new, self._conv_rows[self._oldest])
        ttnn.deallocate(qkv_new)
        self._oldest = (self._oldest + 1) % self.conv_kernel_size

        # 3. Conv1d on device: weighted sum + SiLU
        acc = ttnn.multiply(self._conv_rows[self._oldest], self._conv_w_devs[0])
        for i in range(1, self.conv_kernel_size):
            idx = (self._oldest + i) % self.conv_kernel_size
            product = ttnn.multiply(self._conv_rows[idx], self._conv_w_devs[i])
            old_acc = acc
            acc = ttnn.add(acc, product)
            ttnn.deallocate(product)
            ttnn.deallocate(old_acc)
        conv_out = ttnn.silu(acc)
        ttnn.deallocate(acc)

        # 4. Extract z and b/a from projection (device slice, no host sync)
        z_flat = ttnn.slice(proj, [0, 0, 0, s[0]], [1, 1, B, s[0] + s[1]])
        ba_start = s[0] + s[1]
        ba_flat = ttnn.slice(proj, [0, 0, 0, ba_start], [1, 1, B, ba_start + 2 * self.num_v_heads])
        ttnn.deallocate(proj)

        # 5. Fused kernel
        result = ttnn.experimental.gated_delta_net(
            conv_out,
            z_flat,
            ba_flat,
            self._dt_bias_dev,
            self._neg_A_exp_dev,
            self._dev_state,
            self._norm_w_dev,
            scale=self.scale,
            norm_eps=self.args.norm_eps,
            key_dim=self.key_dim,
            gqa_ratio=self.gqa_ratio,
        )
        output_tt = result[0]
        ttnn.copy(result[1], self._dev_state)
        ttnn.deallocate(result[1])
        ttnn.deallocate(conv_out)
        ttnn.deallocate(z_flat)
        ttnn.deallocate(ba_flat)

        # 6. Reshape kernel output for out_proj: [1, H, B, D] -> [1, 1, B, H*D]
        # For batch>1, permute is needed because ttnn.reshape can't flatten dim1->dim3 in tile layout.
        if output_tt.shape[2] > 1:
            output_tt = ttnn.permute(output_tt, [0, 2, 1, 3])
        output_tt = ttnn.reshape(output_tt, [1, 1, -1, self.value_dim])
        output = ttnn.linear(output_tt, self.out_proj, compute_kernel_config=self.proj_compute_config)
        return output

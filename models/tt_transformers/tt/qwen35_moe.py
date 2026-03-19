# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""
Qwen3.5 MoE (Mixture of Experts) layer for 35B-A3B.

256 experts (top-8 routed) + 1 shared expert per layer.
Router runs on host (softmax + topk). Expert MLPs run on device.
Expert weights stored as bfp4 on DRAM to fit 35B params in 28 GB.
"""

import torch
import torch.nn.functional as F

import ttnn
from models.common.lightweightmodule import LightweightModule


class Qwen35MoE(LightweightModule):
    """MoE layer: host routing + device expert MLPs.

    Each expert: SwiGLU with fused gate+up projection.
      hidden = silu(x @ gate) * (x @ up)  # gate+up fused as single [2048, 1024] weight
      output = hidden @ down

    Shared expert: same SwiGLU but always active, gated by sigmoid.
    """

    def __init__(
        self,
        mesh_device,
        tt_ccl,
        args,
        state_dict,
        weight_cache_path,
        layer_num,
        dtype,
        model_config=None,
        prefetcher=None,
        state_dict_prefix=None,
    ):
        super().__init__()
        self.args = args
        self.mesh_device = mesh_device
        self.layer_num = layer_num

        self.hidden_size = args.dim  # 2048
        self.num_experts = args.num_experts  # 256
        self.num_experts_per_tok = args.num_experts_per_tok  # 8
        self.moe_intermediate_size = args.moe_intermediate_size  # 512
        self.shared_expert_intermediate_size = args.shared_expert_intermediate_size  # 512

        prefix = state_dict_prefix or args.get_state_dict_prefix("Qwen35MoE", layer_num)
        expert_dtype = ttnn.bfloat4_b  # bfp4 to fit in DRAM

        if args.dummy_weights:
            cache_name = lambda _: None
        else:
            cache_name = lambda name: weight_cache_path / f"{prefix}.{name}"

        # Router weight on host (small: [256, 2048])
        self.router_weight = state_dict[f"{prefix}.gate.weight"].float()  # [num_experts, hidden_size]

        # Packed expert weights: gate_up_proj [256, 1024, 2048], down_proj [256, 2048, 512]
        # Store each expert's weights individually on device for selective computation.
        raw_gate_up = state_dict[f"{prefix}.experts.gate_up_proj"]  # [256, 2*moe_intermediate, hidden]
        raw_down = state_dict[f"{prefix}.experts.down_proj"]  # [256, hidden, moe_intermediate]

        self.expert_gate_up = []
        self.expert_down = []
        for e in range(self.num_experts):
            # gate_up: [2*intermediate, hidden] -> transposed [hidden, 2*intermediate] -> [1,1,H,W]
            gu = raw_gate_up[e].T.unsqueeze(0).unsqueeze(0).contiguous()
            self.expert_gate_up.append(
                ttnn.as_tensor(
                    gu,
                    dtype=expert_dtype,
                    device=mesh_device,
                    layout=ttnn.TILE_LAYOUT,
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                    cache_file_name=cache_name(f"experts.{e}.gate_up"),
                )
            )
            # down: [hidden, intermediate] -> transposed [intermediate, hidden] -> [1,1,H,W]
            dn = raw_down[e].T.unsqueeze(0).unsqueeze(0).contiguous()
            self.expert_down.append(
                ttnn.as_tensor(
                    dn,
                    dtype=expert_dtype,
                    device=mesh_device,
                    layout=ttnn.TILE_LAYOUT,
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                    cache_file_name=cache_name(f"experts.{e}.down"),
                )
            )

        # Shared expert weights (always active)
        shared_prefix = f"{prefix}.shared_expert"
        shared_dtype = dtype  # bfp8

        def load_shared(name, shape_transpose=True):
            w = state_dict[f"{shared_prefix}.{name}.weight"]
            if shape_transpose:
                w = w.T
            return ttnn.as_tensor(
                w.unsqueeze(0).unsqueeze(0).contiguous(),
                dtype=shared_dtype,
                device=mesh_device,
                layout=ttnn.TILE_LAYOUT,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                cache_file_name=cache_name(f"shared_expert.{name}"),
            )

        self.shared_w1 = load_shared("gate_proj")  # [1,1,hidden,intermediate]
        self.shared_w3 = load_shared("up_proj")  # [1,1,hidden,intermediate]
        self.shared_w2 = load_shared("down_proj")  # [1,1,intermediate,hidden]

        # Shared expert gate (scalar sigmoid gate)
        self.shared_gate_weight = state_dict[f"{prefix}.shared_expert_gate.weight"].float()  # [1]

    def forward(self, x, mode=None):
        """
        x: [1, 1, B, hidden_size] on device (B=32 padded batch, only row 0 used for decode)

        Returns: [1, 1, B, hidden_size] on device
        """
        # --- Host routing ---
        x_cpu = ttnn.to_torch(x).float()  # [1, 1, B, hidden]
        token_vec = x_cpu[0, 0, 0, : self.hidden_size]  # [hidden]

        # Router logits and top-k selection
        logits = token_vec @ self.router_weight.T  # [num_experts]
        topk_vals, topk_ids = torch.topk(logits, self.num_experts_per_tok)  # [k]
        weights = F.softmax(topk_vals, dim=-1)  # [k] renormalized

        # --- Shared expert (always on) ---
        shared_out = ttnn.linear(x, self.shared_w1, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        shared_up = ttnn.linear(x, self.shared_w3, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        shared_out = ttnn.mul(
            shared_out,
            shared_up,
            input_tensor_a_activations=[ttnn.UnaryOpType.SILU],
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(shared_up)
        shared_out = ttnn.linear(shared_out, self.shared_w2, memory_config=ttnn.DRAM_MEMORY_CONFIG)

        # Apply shared expert gate (sigmoid)
        gate_val = torch.sigmoid(self.shared_gate_weight).item()
        shared_out = ttnn.multiply(shared_out, gate_val, memory_config=ttnn.DRAM_MEMORY_CONFIG)

        # --- Routed experts (top-k, sequential) ---
        result = shared_out
        for i in range(self.num_experts_per_tok):
            eid = topk_ids[i].item()
            w = weights[i].item()

            # Expert SwiGLU: silu(x @ gate) * (x @ up), fused as single gate_up matmul then split
            gate_up_out = ttnn.linear(
                x, self.expert_gate_up[eid], memory_config=ttnn.DRAM_MEMORY_CONFIG
            )  # [1,1,B, 2*intermediate]

            # Split into gate and up halves
            gate_out, up_out = ttnn.split(gate_up_out, 2, dim=-1)
            ttnn.deallocate(gate_up_out)

            hidden = ttnn.mul(
                gate_out,
                up_out,
                input_tensor_a_activations=[ttnn.UnaryOpType.SILU],
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
            ttnn.deallocate(gate_out)
            ttnn.deallocate(up_out)

            expert_out = ttnn.linear(hidden, self.expert_down[eid], memory_config=ttnn.DRAM_MEMORY_CONFIG)
            ttnn.deallocate(hidden)

            # Weighted accumulation
            if w != 1.0:
                expert_out = ttnn.multiply(expert_out, w, memory_config=ttnn.DRAM_MEMORY_CONFIG)
            result = ttnn.add(result, expert_out, memory_config=ttnn.DRAM_MEMORY_CONFIG)
            ttnn.deallocate(expert_out)

        return result

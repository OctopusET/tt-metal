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

        # Router weight on device: [1, 1, hidden, num_experts] for ttnn.linear
        router_w = state_dict[f"{prefix}.gate.weight"].T.unsqueeze(0).unsqueeze(0).contiguous()
        self.router_weight_tt = ttnn.as_tensor(
            router_w,
            dtype=ttnn.bfloat16,
            device=mesh_device,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            cache_file_name=cache_name("gate"),
        )

        # Expert weights: fused gate+up (1 matmul + split instead of 2 matmuls)
        raw_gate_up = state_dict[f"{prefix}.experts.gate_up_proj"]  # [256, 2*intermediate, hidden]
        raw_down = state_dict[f"{prefix}.experts.down_proj"]  # [256, hidden, intermediate]
        intermediate = self.moe_intermediate_size

        self.expert_gate_up = []
        self.expert_down = []
        for e in range(self.num_experts):
            gu = raw_gate_up[e].T.unsqueeze(0).unsqueeze(0).contiguous()  # [1,1,hidden,2*intermediate]
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

        def load_shared(name):
            w = state_dict[f"{shared_prefix}.{name}.weight"]
            return ttnn.as_tensor(
                w.T.unsqueeze(0).unsqueeze(0).contiguous(),
                dtype=shared_dtype,
                device=mesh_device,
                layout=ttnn.TILE_LAYOUT,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                cache_file_name=cache_name(f"shared_expert.{name}"),
            )

        self.shared_w1 = load_shared("gate_proj")  # [1,1,hidden,intermediate]
        self.shared_w3 = load_shared("up_proj")  # [1,1,hidden,intermediate]
        self.shared_w2 = load_shared("down_proj")  # [1,1,intermediate,hidden]

        # Shared expert gate on device: [1, 1, hidden, 1] for ttnn.linear → sigmoid
        gate_w = state_dict[f"{prefix}.shared_expert_gate.weight"].T.unsqueeze(0).unsqueeze(0).contiguous()
        self.shared_gate_weight_tt = ttnn.as_tensor(
            gate_w,
            dtype=ttnn.bfloat16,
            device=mesh_device,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            cache_file_name=cache_name("shared_expert_gate"),
        )

    def forward(self, x, mode=None):
        """
        x: [1, 1, B, hidden_size] on device (B=32 padded batch, only row 0 used for decode)
        Returns: [1, 1, B, hidden_size] on device
        """
        L1 = ttnn.L1_MEMORY_CONFIG  # Small decode tensors fit in L1, avoid DRAM roundtrips

        # --- Queue shared expert + routing + gate on device BEFORE sync ---
        shared_out = ttnn.linear(x, self.shared_w1, memory_config=L1)
        shared_up = ttnn.linear(x, self.shared_w3, memory_config=L1)
        shared_out = ttnn.mul(
            shared_out,
            shared_up,
            input_tensor_a_activations=[ttnn.UnaryOpType.SILU],
            memory_config=L1,
        )
        ttnn.deallocate(shared_up)
        shared_out = ttnn.linear(shared_out, self.shared_w2, memory_config=L1)

        # Router logits on device: [1,1,B,256] -- only sync 256 floats, not 2048
        router_logits = ttnn.linear(x, self.router_weight_tt, memory_config=L1)

        # Shared expert gate on device: sigmoid(x @ gate_weight) -> [1,1,B,1]
        gate = ttnn.linear(x, self.shared_gate_weight_tt, memory_config=L1)
        gate = ttnn.sigmoid(gate, memory_config=L1)
        shared_out = ttnn.mul(shared_out, gate, memory_config=L1)
        ttnn.deallocate(gate)

        # --- Sync: read only 256-float router logits (1 KB) ---
        logits_cpu = ttnn.to_torch(router_logits).float()[0, 0, 0, : self.num_experts]
        ttnn.deallocate(router_logits)
        topk_vals, topk_ids = torch.topk(logits_cpu, self.num_experts_per_tok)
        weights = F.softmax(topk_vals, dim=-1)

        # --- Routed experts (top-k, fused gate+up, L1 intermediates) ---
        result = shared_out
        for i in range(self.num_experts_per_tok):
            eid = topk_ids[i].item()
            w = weights[i].item()

            # Fused gate+up: 1 matmul -> split -> SwiGLU (saves 1 matmul dispatch per expert)
            gate_up = ttnn.linear(x, self.expert_gate_up[eid], memory_config=L1)
            gate_out, up_out = ttnn.split(gate_up, self.moe_intermediate_size, dim=3)
            ttnn.deallocate(gate_up)
            hidden = ttnn.mul(
                gate_out,
                up_out,
                input_tensor_a_activations=[ttnn.UnaryOpType.SILU],
                memory_config=L1,
            )
            ttnn.deallocate(gate_out)
            ttnn.deallocate(up_out)

            expert_out = ttnn.linear(hidden, self.expert_down[eid], memory_config=L1)
            ttnn.deallocate(hidden)

            # Weighted accumulation
            if w != 1.0:
                expert_out = ttnn.multiply(expert_out, w, memory_config=L1)
            result = ttnn.add(result, expert_out, memory_config=L1)
            ttnn.deallocate(expert_out)

        return result

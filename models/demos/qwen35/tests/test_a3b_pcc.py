# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
PCC validation tests for Qwen3.5-35B-A3B DeltaNet and MoE layers.

Tests single-layer output against PyTorch reference to verify numerical
correctness before and after optimization changes.
"""

import math
import os

import pytest
import torch
import torch.nn.functional as F

import ttnn

# ---------------------------------------------------------------------------
# Constants (A3B architecture)
# ---------------------------------------------------------------------------
A3B_MODEL = os.environ.get("HF_MODEL", "Qwen/Qwen3.5-35B-A3B")
HIDDEN_SIZE = 2048
NUM_V_HEADS = 32
NUM_K_HEADS = 16
HEAD_K_DIM = 128
HEAD_V_DIM = 128
KEY_DIM = HEAD_K_DIM * NUM_K_HEADS  # 2048
VALUE_DIM = HEAD_V_DIM * NUM_V_HEADS  # 4096
CONV_DIM = KEY_DIM * 2 + VALUE_DIM  # 8192
CONV_KERNEL_SIZE = 4
GQA_RATIO = NUM_V_HEADS // NUM_K_HEADS  # 2
MOE_INTERMEDIATE = 512
NUM_EXPERTS = 256
NUM_EXPERTS_PER_TOK = 8
NORM_EPS = 1e-6
PCC_THRESHOLD = 0.99


def compute_pcc(x: torch.Tensor, y: torch.Tensor) -> float:
    x_flat = x.flatten().float()
    y_flat = y.flatten().float()
    x_c = x_flat - x_flat.mean()
    y_c = y_flat - y_flat.mean()
    num = (x_c * y_c).sum()
    den = torch.sqrt((x_c**2).sum() * (y_c**2).sum())
    return (num / den).item() if den > 0 else 0.0


def l2norm(x, dim=-1, eps=1e-6):
    return x * torch.rsqrt((x * x).sum(dim=dim, keepdim=True) + eps)


def resolve_model_path():
    path = A3B_MODEL
    if not os.path.isdir(path):
        from huggingface_hub import snapshot_download

        path = snapshot_download(path)
    return path


def load_layer_weights(model_path, layer_idx):
    import json

    from safetensors.torch import safe_open

    index_path = os.path.join(model_path, "model.safetensors.index.json")
    with open(index_path) as f:
        index = json.load(f)

    prefix = f"model.language_model.layers.{layer_idx}."
    needed_files = set()
    for key, shard_file in index["weight_map"].items():
        if key.startswith(prefix):
            needed_files.add(shard_file)

    weights = {}
    for shard_file in needed_files:
        shard_path = os.path.join(model_path, shard_file)
        with safe_open(shard_path, framework="pt", device="cpu") as f:
            for key in f.keys():
                if key.startswith(prefix):
                    weights[key] = f.get_tensor(key)
    return weights


# ---------------------------------------------------------------------------
# Reference: DeltaNet single-step (PyTorch, float32)
# ---------------------------------------------------------------------------
def ref_deltanet_forward(weights, hidden_states, layer_idx=0, prev_state=None, prev_conv_state=None):
    """
    Single-token DeltaNet forward with optional persistent state.

    Returns: (output, new_state, new_conv_state)
    """
    p = f"model.language_model.layers.{layer_idx}."

    in_proj_qkv_w = weights[f"{p}linear_attn.in_proj_qkv.weight"]
    in_proj_z_w = weights[f"{p}linear_attn.in_proj_z.weight"]
    in_proj_b_w = weights[f"{p}linear_attn.in_proj_b.weight"]
    in_proj_a_w = weights[f"{p}linear_attn.in_proj_a.weight"]
    out_proj_w = weights[f"{p}linear_attn.out_proj.weight"]
    conv1d_w = weights[f"{p}linear_attn.conv1d.weight"].squeeze(1)  # (conv_dim, kernel)
    dt_bias = weights[f"{p}linear_attn.dt_bias"]
    A_log = weights[f"{p}linear_attn.A_log"]
    norm_w = weights[f"{p}linear_attn.norm.weight"]

    batch_size = hidden_states.shape[0]

    # Projections
    mixed_qkv = F.linear(hidden_states, in_proj_qkv_w)
    z = F.linear(hidden_states, in_proj_z_w)
    b = F.linear(hidden_states, in_proj_b_w)
    a = F.linear(hidden_states, in_proj_a_w)

    # Conv1d ring buffer
    if prev_conv_state is None:
        conv_state = torch.zeros(CONV_KERNEL_SIZE, CONV_DIM)
    else:
        conv_state = prev_conv_state.clone()
    conv_state[:-1] = conv_state[1:].clone()
    conv_state[-1] = mixed_qkv[0, 0, :]
    conv_w_t = conv1d_w.T  # (kernel, conv_dim)
    mixed_qkv = F.silu((conv_state * conv_w_t).sum(dim=0))

    # Split + reshape + GQA + L2 norm
    q = mixed_qkv[:KEY_DIM].reshape(NUM_K_HEADS, HEAD_K_DIM)
    k = mixed_qkv[KEY_DIM : 2 * KEY_DIM].reshape(NUM_K_HEADS, HEAD_K_DIM)
    v = mixed_qkv[2 * KEY_DIM :].reshape(NUM_V_HEADS, HEAD_V_DIM)
    q = q.repeat_interleave(GQA_RATIO, dim=0)
    k = k.repeat_interleave(GQA_RATIO, dim=0)
    scale = 1.0 / math.sqrt(HEAD_K_DIM)
    q = F.normalize(q, dim=-1) * scale
    k = F.normalize(k, dim=-1)

    # Gates
    beta = b[0, 0, :].sigmoid()
    A_exp = A_log.float().exp()
    decay = (-A_exp * F.softplus(a[0, 0, :].float() + dt_bias)).exp()

    # Recurrence
    if prev_state is None:
        state = torch.zeros(NUM_V_HEADS, HEAD_K_DIM, HEAD_V_DIM)
    else:
        state = prev_state.clone()
    state *= decay.unsqueeze(-1).unsqueeze(-1)
    kv_mem = torch.bmm(k.unsqueeze(1), state).squeeze(1)
    delta = (v - kv_mem) * beta.unsqueeze(-1)
    state += torch.bmm(k.unsqueeze(2), delta.unsqueeze(1))
    output_h = torch.bmm(q.unsqueeze(1), state).squeeze(1)

    # Gated RMSNorm
    z_heads = z[0, 0, :].reshape(NUM_V_HEADS, HEAD_V_DIM)
    variance = output_h.pow(2).mean(-1, keepdim=True)
    output_normed = output_h * torch.rsqrt(variance + NORM_EPS)
    output_normed = output_normed * norm_w
    output_gated = output_normed * F.silu(z_heads)
    output_flat = output_gated.reshape(1, 1, VALUE_DIM)

    # Output projection
    output = F.linear(output_flat, out_proj_w)
    return output, state, conv_state


# ---------------------------------------------------------------------------
# Reference: MoE single-step (PyTorch, float32)
# ---------------------------------------------------------------------------
def ref_moe_forward(weights, hidden_states, layer_idx=0):
    """Single-token MoE forward."""
    p = f"model.language_model.layers.{layer_idx}."
    token = hidden_states[0, 0, :]

    # Router
    router_w = weights[f"{p}mlp.gate.weight"]
    logits = token @ router_w.T
    topk_vals, topk_ids = torch.topk(logits, NUM_EXPERTS_PER_TOK)
    routing_weights = F.softmax(topk_vals, dim=-1)

    # Shared expert
    shared_gate_w = weights[f"{p}mlp.shared_expert.gate_proj.weight"]
    shared_up_w = weights[f"{p}mlp.shared_expert.up_proj.weight"]
    shared_down_w = weights[f"{p}mlp.shared_expert.down_proj.weight"]
    shared_gate_out = F.silu(F.linear(hidden_states, shared_gate_w))
    shared_up_out = F.linear(hidden_states, shared_up_w)
    shared_out = F.linear(shared_gate_out * shared_up_out, shared_down_w)

    # Shared expert gate
    shared_gate_scalar_w = weights[f"{p}mlp.shared_expert_gate.weight"]
    gate_val = torch.sigmoid(token @ shared_gate_scalar_w.T)
    shared_out = shared_out * gate_val

    # Routed experts
    gate_up = weights[f"{p}mlp.experts.gate_up_proj"]  # [256, 1024, 2048]
    down = weights[f"{p}mlp.experts.down_proj"]  # [256, 2048, 512]

    routed_out = torch.zeros_like(hidden_states)
    for i in range(NUM_EXPERTS_PER_TOK):
        eid = topk_ids[i].item()
        w = routing_weights[i].item()
        gate_w = gate_up[eid, :MOE_INTERMEDIATE, :]  # [512, 2048]
        up_w = gate_up[eid, MOE_INTERMEDIATE:, :]  # [512, 2048]
        down_w = down[eid]  # [2048, 512]
        gate_out = F.silu(F.linear(hidden_states, gate_w))
        up_out = F.linear(hidden_states, up_w)
        expert_out = F.linear(gate_out * up_out, down_w)
        routed_out += w * expert_out

    return shared_out + routed_out


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def model_path():
    return resolve_model_path()


@pytest.fixture(scope="module")
def device():
    dev = ttnn.open_device(device_id=0)
    yield dev
    ttnn.close_device(dev)


@pytest.fixture(scope="module")
def model_args(device, model_path):
    os.environ["HF_MODEL"] = model_path
    from models.tt_transformers.tt.model_config import ModelArgs

    return ModelArgs(device, max_seq_len=256)


class TestDeltaNetPCC:
    """PCC tests for DeltaNet (linear attention) layers."""

    def test_single_step(self, device, model_path, model_args):
        """Single DeltaNet layer, single token: PCC >= 0.99."""
        weights = load_layer_weights(model_path, 0)
        torch.manual_seed(42)
        x = torch.randn(1, 1, HIDDEN_SIZE)

        # Reference
        ref_out, _, _ = ref_deltanet_forward(weights, x, layer_idx=0)

        # TTNN
        sd = model_args.load_state_dict()
        wcp = model_args.weight_cache_path(dtype=ttnn.bfloat8_b)
        from models.tt_transformers.tt.gated_deltanet import GatedDeltaNet

        layer = GatedDeltaNet(
            mesh_device=device,
            args=model_args,
            state_dict=sd,
            weight_cache_path=wcp,
            layer_num=0,
            dtype=ttnn.bfloat8_b,
        )
        layer.initialize_states()

        B = model_args.tile_padded_batch_rows
        x_pad = torch.zeros(1, 1, B, HIDDEN_SIZE)
        x_pad[0, 0, 0, :] = x[0, 0, :]
        x_tt = ttnn.from_torch(x_pad, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
        out_tt = layer.forward(x_tt)
        out_cpu = ttnn.to_torch(out_tt).float()[0, 0, 0, :HIDDEN_SIZE]

        pcc = compute_pcc(ref_out[0, 0, :], out_cpu)
        print(f"DeltaNet single-step PCC: {pcc:.6f}")
        assert pcc >= PCC_THRESHOLD, f"DeltaNet PCC {pcc:.4f} < {PCC_THRESHOLD}"

    def test_multi_step(self, device, model_path, model_args):
        """10 sequential tokens through one DeltaNet layer: verify state PCC."""
        weights = load_layer_weights(model_path, 0)
        torch.manual_seed(42)

        sd = model_args.load_state_dict()
        wcp = model_args.weight_cache_path(dtype=ttnn.bfloat8_b)
        from models.tt_transformers.tt.gated_deltanet import GatedDeltaNet

        layer = GatedDeltaNet(
            mesh_device=device,
            args=model_args,
            state_dict=sd,
            weight_cache_path=wcp,
            layer_num=0,
            dtype=ttnn.bfloat8_b,
        )
        layer.initialize_states()

        B = model_args.tile_padded_batch_rows
        ref_state, ref_conv_state = None, None
        pccs = []

        for step in range(10):
            x = torch.randn(1, 1, HIDDEN_SIZE)

            # Reference
            ref_out, ref_state, ref_conv_state = ref_deltanet_forward(
                weights, x, layer_idx=0, prev_state=ref_state, prev_conv_state=ref_conv_state
            )

            # TTNN
            x_pad = torch.zeros(1, 1, B, HIDDEN_SIZE)
            x_pad[0, 0, 0, :] = x[0, 0, :]
            x_tt = ttnn.from_torch(x_pad, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
            out_tt = layer.forward(x_tt)
            out_cpu = ttnn.to_torch(out_tt).float()[0, 0, 0, :HIDDEN_SIZE]

            pcc = compute_pcc(ref_out[0, 0, :], out_cpu)
            pccs.append(pcc)
            print(f"  Step {step}: PCC = {pcc:.6f}")

        min_pcc = min(pccs)
        print(f"DeltaNet multi-step min PCC: {min_pcc:.6f}")
        assert min_pcc >= 0.95, f"DeltaNet multi-step min PCC {min_pcc:.4f} < 0.95"


class TestMoEPCC:
    """PCC tests for MoE (Mixture of Experts) layers."""

    def test_single_layer(self, device, model_path, model_args):
        """Single MoE layer: PCC >= 0.99."""
        weights = load_layer_weights(model_path, 0)
        torch.manual_seed(42)
        x = torch.randn(1, 1, HIDDEN_SIZE)

        # Reference
        ref_out = ref_moe_forward(weights, x, layer_idx=0)

        # TTNN
        sd = model_args.load_state_dict()
        wcp = model_args.weight_cache_path(dtype=ttnn.bfloat8_b)
        from models.tt_transformers.tt.qwen35_moe import Qwen35MoE

        moe = Qwen35MoE(
            mesh_device=device,
            tt_ccl=None,
            args=model_args,
            state_dict=sd,
            weight_cache_path=wcp,
            layer_num=0,
            dtype=ttnn.bfloat8_b,
        )

        B = model_args.tile_padded_batch_rows
        x_pad = torch.zeros(1, 1, B, HIDDEN_SIZE)
        x_pad[0, 0, 0, :] = x[0, 0, :]
        x_tt = ttnn.from_torch(x_pad, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
        out_tt = moe.forward(x_tt)
        out_cpu = ttnn.to_torch(out_tt).float()[0, 0, 0, :HIDDEN_SIZE]

        pcc = compute_pcc(ref_out[0, 0, :], out_cpu)
        print(f"MoE single-layer PCC: {pcc:.6f}")
        assert pcc >= PCC_THRESHOLD, f"MoE PCC {pcc:.4f} < {PCC_THRESHOLD}"


class TestFusedKernelPCC:
    """PCC tests for the fused DeltaNet Metalium kernel."""

    def test_single_step(self, device):
        """Fused kernel single step: PCC >= 0.999."""
        H, D = 32, 128  # A3B DeltaNet config
        torch.manual_seed(42)
        q_t = torch.randn(1, H, 1, D)
        k_t = torch.randn(1, H, 1, D)
        v_t = torch.randn(1, H, 1, D)
        decay_t = torch.rand(1, H, 1, 1) * 0.5 + 0.5
        beta_t = torch.sigmoid(torch.randn(1, H, 1, 1))
        state_t = torch.randn(1, H, D, D) * 0.1

        # Reference
        ref_state = state_t.clone() * decay_t
        kv_mem = torch.matmul(k_t, ref_state)
        delta = (v_t - kv_mem) * beta_t
        ref_state += torch.matmul(k_t.transpose(-2, -1), delta)
        ref_output = torch.matmul(q_t, ref_state)

        # Device (k as column vector, v as pre-computed delta)
        k_col = k_t.transpose(-2, -1)
        q = ttnn.from_torch(q_t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
        k = ttnn.from_torch(k_col, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
        v = ttnn.from_torch(delta, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
        decay = ttnn.from_torch(decay_t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
        beta = ttnn.from_torch(torch.ones(1, H, 1, 1), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
        state = ttnn.from_torch(state_t, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)

        output, new_state = ttnn.experimental.gated_delta_net(q, k, v, decay, beta, state)
        out_cpu = ttnn.to_torch(output).float()
        state_cpu = ttnn.to_torch(new_state).float()

        pcc_out = compute_pcc(out_cpu, ref_output)
        pcc_state = compute_pcc(state_cpu, ref_state)
        print(f"Fused kernel Output PCC: {pcc_out:.6f}")
        print(f"Fused kernel State PCC:  {pcc_state:.6f}")
        assert pcc_out >= 0.999, f"Fused kernel output PCC {pcc_out:.4f} < 0.999"
        assert pcc_state >= 0.999, f"Fused kernel state PCC {pcc_state:.4f} < 0.999"

    def test_multi_step(self, device):
        """Fused kernel 10 sequential steps: state PCC stays > 0.99."""
        H, D = 32, 128
        torch.manual_seed(42)
        ref_state = torch.zeros(1, H, D, D)
        dev_state = ttnn.from_torch(ref_state, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)

        pccs = []
        for step in range(10):
            q_t = torch.randn(1, H, 1, D) * 0.1
            k_t = torch.randn(1, H, 1, D) * 0.1
            v_t = torch.randn(1, H, 1, D) * 0.1
            decay_t = torch.rand(1, H, 1, 1) * 0.3 + 0.7
            beta_t = torch.sigmoid(torch.randn(1, H, 1, 1))

            # Reference
            ref_state = ref_state * decay_t
            kv_mem = torch.matmul(k_t, ref_state)
            delta = (v_t - kv_mem) * beta_t
            ref_state = ref_state + torch.matmul(k_t.transpose(-2, -1), delta)

            # Device
            k_col = k_t.transpose(-2, -1)
            q = ttnn.from_torch(q_t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
            k = ttnn.from_torch(k_col, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
            v = ttnn.from_torch(delta, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
            decay = ttnn.from_torch(decay_t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
            beta = ttnn.from_torch(torch.ones(1, H, 1, 1), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
            _, dev_state = ttnn.experimental.gated_delta_net(q, k, v, decay, beta, dev_state)

            state_cpu = ttnn.to_torch(dev_state).float()
            pcc = compute_pcc(state_cpu, ref_state)
            pccs.append(pcc)
            print(f"  Step {step}: state PCC = {pcc:.6f}")

        min_pcc = min(pccs)
        print(f"Fused kernel multi-step min state PCC: {min_pcc:.6f}")
        assert min_pcc >= 0.99, f"Fused kernel multi-step min PCC {min_pcc:.4f} < 0.99"

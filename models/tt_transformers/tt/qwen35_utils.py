# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""Qwen3.5-specific weight conversion utilities."""

from models.tt_transformers.tt.load_checkpoints import map_hf_to_meta_keys, split_hf_keys


def _is_moe_key(key):
    """Check if a key belongs to MoE-specific weights that need protection from transforms."""
    return any(pat in key for pat in ("mlp.experts", "mlp.gate.", "mlp.shared_expert"))


def convert_hf_to_meta_qwen35(state_dict, head_dim, n_heads, n_kv_heads):
    """
    Convert Qwen3.5 HF checkpoint to meta format.

    Qwen3.5 has two layer types:
    - linear_attention layers: use GatedDeltaNet with keys like linear_attn.in_proj_qkv, etc.
      These keys are kept as-is (no meta format conversion needed for linear_attn weights).
    - full_attention layers: standard GQA with a gated q_proj (2x normal size, half is gate).
      The q_proj must be split into wq (query) and wq_gate before reverse_permute.

    MoE variant (Qwen3.5-35B-A3B):
    - Expert weights are packed 3D tensors (experts.gate_up_proj [256,1024,2048],
      experts.down_proj [256,2048,512]) that must NOT be split by split_hf_keys.
    - MoE keys must NOT go through map_hf_to_meta_keys (would rename gate_proj->w1
      inside expert paths). Only mlp->feed_forward rename is applied.
    """
    # Pop MoE keys before transforms to protect them
    moe_keys = {k: v for k, v in state_dict.items() if _is_moe_key(k)}
    if moe_keys:
        state_dict = {k: v for k, v in state_dict.items() if not _is_moe_key(k)}

    # Step 1: Split fused keys (gate_up_proj, qkv_proj) if any
    state_dict = split_hf_keys(state_dict, n_heads, n_kv_heads)

    # Step 2: Handle the 2x q_proj for full_attention layers (contains query + gate)
    # and apply reverse_permute to q/k weights. Skip linear_attn keys entirely.
    converted_weights = {}
    for key, tensor in state_dict.items():
        if "linear_attn" in key:
            # DeltaNet layer weights: pass through unchanged
            converted_weights[key] = tensor
        elif "q_proj.weight" in key:
            # Qwen3.5 full_attention q_proj is [n_heads * head_dim * 2, hidden_size]
            # First half is query, second half is gate.
            # NO reverse_permute: Qwen3.5 uses HF-style RoPE with partial_rotary_factor,
            # so weights are already in the correct format.
            q_size = n_heads * head_dim
            if tensor.shape[0] == q_size * 2:
                # Weight layout is interleaved per-head: [Q_h0(hd), G_h0(hd), Q_h1(hd), G_h1(hd), ...]
                # Split per-head into query and gate (NOT first/second half!)
                w = tensor.reshape(n_heads, head_dim * 2, -1)
                converted_weights[key] = w[:, :head_dim, :].reshape(q_size, -1).contiguous()
                gate_key = key.replace("q_proj.weight", "q_proj_gate.weight")
                converted_weights[gate_key] = w[:, head_dim:, :].reshape(q_size, -1).contiguous()
            else:
                converted_weights[key] = tensor
        elif "k_proj.weight" in key:
            converted_weights[key] = tensor
        elif "q_proj.bias" in key:
            q_size = n_heads * head_dim
            if tensor.shape[0] == q_size * 2:
                # Same interleaved per-head split as weights
                b = tensor.reshape(n_heads, head_dim * 2)
                converted_weights[key] = b[:, :head_dim].reshape(-1).contiguous()
                gate_key = key.replace("q_proj.bias", "q_proj_gate.bias")
                converted_weights[gate_key] = b[:, head_dim:].reshape(-1).contiguous()
            else:
                converted_weights[key] = tensor
        elif "k_proj.bias" in key:
            converted_weights[key] = tensor
        elif "q_norm.weight" in key:
            converted_weights[key] = tensor  # No permute for HF-style RoPE
        elif "k_norm.weight" in key:
            converted_weights[key] = tensor
        else:
            converted_weights[key] = tensor

    # Step 3: Map HF keys to meta keys (linear_attn keys pass through since they
    # don't match any replacement patterns like self_attn, q_proj, etc.)
    converted_weights = map_hf_to_meta_keys(converted_weights)

    # Step 4: Re-insert MoE keys with only mlp->feed_forward rename
    for key, tensor in moe_keys.items():
        new_key = key.replace(".mlp.", ".feed_forward.")
        converted_weights[new_key] = tensor

    return converted_weights

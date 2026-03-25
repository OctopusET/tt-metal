#!/usr/bin/env python3
"""Test fused GatedDeltaNet kernel with flat tensor API (conv_out, z_flat, ba_flat)."""
import math

import torch
import ttnn

device = ttnn.open_device(device_id=0)

H, D, B = 32, 128, 32
KEY_DIM = 2048
CONV_DIM = 8192
NORM_EPS = 1e-6
SCALE = 1.0 / math.sqrt(D)
GQA = 2

torch.manual_seed(42)
F = torch.nn.functional

# Create flat conv output: (1, 1, B, conv_dim) with q|k|v in row 0
q_raw = torch.randn(H // GQA, D)  # 16 k-heads
k_raw = torch.randn(H // GQA, D)
v_raw = torch.randn(H, D)  # 32 v-heads
conv_flat = torch.zeros(CONV_DIM)
conv_flat[:KEY_DIM] = q_raw.reshape(-1)
conv_flat[KEY_DIM : 2 * KEY_DIM] = k_raw.reshape(-1)
conv_flat[2 * KEY_DIM :] = v_raw.reshape(-1)
conv_out_t = torch.zeros(1, 1, B, CONV_DIM, dtype=torch.bfloat16)
conv_out_t[0, 0, 0, :] = conv_flat.bfloat16()

# z, b, a as flat tensors
z_raw = torch.randn(H, D)
z_flat_t = torch.zeros(1, 1, B, H * D, dtype=torch.bfloat16)
z_flat_t[0, 0, 0, :] = z_raw.reshape(-1).bfloat16()

b_raw = torch.randn(H)
a_raw = torch.randn(H)
ba_flat_t = torch.zeros(1, 1, B, 2 * H, dtype=torch.bfloat16)
ba_flat_t[0, 0, 0, :H] = b_raw.bfloat16()
ba_flat_t[0, 0, 0, H:] = a_raw.bfloat16()

dt_bias_t = torch.randn(1, H, 1, 1, dtype=torch.bfloat16)
neg_A_exp_t = -torch.rand(1, H, 1, 1).abs().to(torch.bfloat16)
state_t = torch.randn(1, H, D, D, dtype=torch.float32) * 0.1
norm_w_t = torch.randn(1, H, 1, D, dtype=torch.bfloat16).abs() + 0.5

# Reference
q_h = q_raw.repeat_interleave(GQA, dim=0)
k_h = k_raw.repeat_interleave(GQA, dim=0)
q_f = F.normalize(q_h.float(), dim=-1).unsqueeze(0).unsqueeze(2) * SCALE
k_f = F.normalize(k_h.float(), dim=-1).unsqueeze(0).unsqueeze(2)
v_f = v_raw.float().unsqueeze(0).unsqueeze(2)
z_f = z_raw.float().unsqueeze(0).unsqueeze(2)
beta = torch.sigmoid(b_raw.reshape(1, H, 1, 1).float())
sp = F.softplus(a_raw.reshape(1, H, 1, 1).float() + dt_bias_t.float())
decay = torch.exp(neg_A_exp_t.float() * sp)

ref_state = state_t.clone() * decay
kv_mem = k_f @ ref_state
delta = (v_f - kv_mem) * beta
ref_state = ref_state + k_f.transpose(-2, -1) @ delta
raw_out = q_f @ ref_state
var = raw_out.pow(2).mean(-1, keepdim=True)
normed = raw_out * torch.rsqrt(var + NORM_EPS) * norm_w_t.float()
ref_out = (normed * F.silu(z_f)).to(torch.bfloat16)

to_dev = lambda t: ttnn.from_torch(t, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG)

result = ttnn.experimental.gated_delta_net(
    to_dev(conv_out_t),
    to_dev(z_flat_t),
    to_dev(ba_flat_t),
    to_dev(dt_bias_t),
    to_dev(neg_A_exp_t),
    to_dev(state_t),
    to_dev(norm_w_t),
    scale=SCALE,
    norm_eps=NORM_EPS,
    key_dim=KEY_DIM,
    gqa_ratio=GQA,
)
ttnn.synchronize_device(device)

out_tt = ttnn.to_torch(result[0])
state_tt = ttnn.to_torch(result[1])

pcc_out = torch.nn.functional.cosine_similarity(
    out_tt[:, :H, :1, :D].float().flatten(), ref_out.float().flatten(), dim=0
).item()
pcc_state = torch.nn.functional.cosine_similarity(
    state_tt[:, :H, :D, :D].float().flatten(), ref_state.float().flatten(), dim=0
).item()

print(f"output PCC:    {pcc_out:.6f}", flush=True)
print(f"state_new PCC: {pcc_state:.6f}", flush=True)
assert pcc_out >= 0.998, f"Output PCC {pcc_out} < 0.998"
assert pcc_state >= 0.999, f"State PCC {pcc_state} < 0.999"
print("ALL PASS", flush=True)

ttnn.close_device(device)

# Qwen3.5 Performance

Performance collected on single P100A Blackhole.

## Decode Performance

| Model | Device | Precision | Speed (tok/s) | Demo |
|---|---|---|---|---|
| Qwen3.5-35B-A3B | P100A | bfp4 experts, f32 recurrence | 11.7 | [demo_a3b.py](demo/demo_a3b.py) |
| Qwen3.5-27B | P100A | bfp8 weights, f32 recurrence | 6.28 | [demo.py](demo/demo.py) |

## Profiling (A3B, 86ms/token)

| Component | Time | Syncs | Notes |
|---|---|---|---|
| DeltaNet (30 layers) | 54ms | 30 | Host recurrence, 1 sync/layer |
| Attention (10 layers) | 18ms | 10+40 | Device RoPE, HfRotarySetup |
| norm + LM head | 14ms | 1 | Host embedding (248K vocab) |
| **Total** | **86ms** | **~70** | |

Breakdown: ~35ms sync + ~26ms Python dispatch + ~20ms device compute.

## DRAM Usage

### A3B (MoE)

| Component | Dtype | Size |
|---|---|---|
| Expert weights (256 x gate+up+down) | bfp4 | 12.8 GB |
| Shared expert weights (40 layers) | bfp8 | 0.8 GB |
| DeltaNet projections (30 layers) | bfp8 | 1.2 GB |
| Attention QKV+WO+gate (10 layers) | bf16 | 0.5 GB |
| Router + shared gate weights | bf16 | 0.1 GB |
| KV cache (10 layers) | bf16 | 0.3 GB |
| **Total** | | **~15.7 GB / 28 GB** |

### 27B (Dense)

| Component | Dtype | Size |
|---|---|---|
| DeltaNet projections (48 layers) | bfp8 | 5.4 GB |
| MLP w1+w2+w3 (64 layers) | bfp8 | 17.1 GB |
| Attention QKV+WO+gate (16 layers) | bf16 | 2.2 GB |
| Other | bf16 | ~0.3 GB |
| **Total** | | **~25 GB / 28 GB** |

## Known Bottlenecks

- DeltaNet host recurrence: 1 `to_torch` + `from_torch` sync per layer per token
- bf16 element-wise ops on Blackhole fp32 state: SrcB is 19-bit TF32, prevents all-device recurrence
- Host embedding + CPU LM head: 248K vocab too large for device
- MoE expert routing: topk on host (256 floats synced per layer)

## Optimization Opportunities

- Metal Trace for device ops (eliminates Python dispatch overhead)
- Multi-CQ overlap (CQ1 for from_torch writes during CQ0 compute)
- Fused DeltaNet kernel ready (PCC 0.999997), blocked by from_torch overhead without Metal Trace

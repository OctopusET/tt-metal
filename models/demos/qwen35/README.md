# Qwen3.5 on Tenstorrent Blackhole P100A

Qwen3.5-27B (dense) and Qwen3.5-35B-A3B (MoE) decode on single P100A Blackhole ($999).

## Results

| Model | Active params | tok/s | Method |
|-------|-------------|-------|--------|
| Qwen3.5-35B-A3B | 3B | **10.9** | Host recurrence + device MoE |
| Qwen3.5-27B | 27B | **6.28** | Host recurrence + device MLP |

Comparison: AmpereOne 128-core CPU runs A3B at 9.05 tok/s (llama.cpp Q4_K).

## Quick Start

```bash
# A3B (MoE, recommended)
huggingface-cli download Qwen/Qwen3.5-35B-A3B
HF_MODEL=Qwen/Qwen3.5-35B-A3B python models/demos/qwen35/demo/demo_a3b.py \
  --prompt "What is the capital of France?" --max_tokens 80

# 27B (dense)
huggingface-cli download Qwen/Qwen3.5-27B
HF_MODEL=Qwen/Qwen3.5-27B python models/demos/qwen35/demo/demo.py \
  --prompt "What is the capital of France?" --max_tokens 80
```

## Tests

```bash
# Fused kernel PCC (no model download needed)
python -m pytest models/demos/qwen35/tests/test_a3b_pcc.py::TestFusedKernelPCC -v -s

# Full PCC tests (needs model)
HF_MODEL=Qwen/Qwen3.5-35B-A3B python -m pytest models/demos/qwen35/tests/test_a3b_pcc.py -v -s
HF_MODEL=Qwen/Qwen3.5-27B python -m pytest models/demos/qwen35/tests/test_pcc.py -v -s
```

## Architecture

### Models

| | 27B | 35B-A3B | 122B-A10B | 397B-A17B |
|---|---|---|---|---|
| Layers | 64 (48+16) | 40 (30+10) | 48 (36+12) | 60 (45+15) |
| Hidden | 5120 | 2048 | 3072 | 4096 |
| DeltaNet V heads | 48 | 32 | 64 | 64 |
| Attn Q/KV heads | 24/4 | 16/2 | 32/2 | 32/2 |
| MLP | Dense SwiGLU | MoE 256 top-8 | MoE 256 top-8 | MoE 512 top-10 |
| Active params | 27B | 3B | 10B | 17B |
| DRAM (bfp4) | ~13 GB | ~17.5 GB | ~61 GB | ~199 GB |
| Fits on | P100A | **P100A** | Galaxy 6U | Galaxy 6U+ |

### DeltaNet Recurrence

```
state *= exp(gate)                          # decay
kv_mem = einsum('hkv,hk->hv', state, k)    # retrieve
delta  = (v - kv_mem) * beta                # correction
state += einsum('hk,hv->hkv', k, delta)    # rank-1 update
output = einsum('hkv,hk->hv', state, q)    # read
```

Runs on host in float32. bf16 compounds over 30+ layers producing garbage output.

### MoE

256 experts (top-8) + 1 shared (always active). Expert weights at bfp4 on device DRAM.
Router matmul on device, topk/softmax on host. Shared expert overlaps with routing sync.

## Profiling (A3B, 86ms/token)

| Component | Time | Syncs |
|-----------|------|-------|
| DeltaNet (30 layers) | 54ms | 30 |
| Attention (10 layers) | 18ms | 10+40 |
| norm + LM head | 14ms | 1 |
| **Total** | **86ms** | **~70** |

Breakdown: ~35ms sync + ~26ms Python dispatch + ~20ms device compute.
Theoretical limit: ~5.8ms/token (172 tok/s). Current efficiency: 6.3%.

## Fused DeltaNet Kernel

Custom Metalium kernel: full recurrence in 1 launch, fp32 state on device DRAM.
PCC 0.999997 vs host float32. Not used in production (from_torch overhead).
Becomes viable with Metal Trace.

See `ttnn/cpp/ttnn/operations/experimental/ssm/gated_delta_net/device/`.

## Blackhole fp32 CB Reference

SrcB register is 19-bit (TF32). Element-wise ops with fp32 input CBs hang.
Workarounds: init_sfpu+copy_tile for fp32 read, SFPU binary path for fp32 ops,
binary_dest_reuse_tiles for mixed fp32-DST + bf16-CB operations.
matmul_tiles has a separate unpack path that handles fp32 CBs.

## File Structure

```
models/tt_transformers/tt/
  gated_deltanet.py         # DeltaNet (host recurrence, device projections)
  gated_attention.py        # GatedAttention (gate + partial RoPE)
  qwen35_moe.py             # MoE (device routing, bfp4 experts)
  qwen35_decoder.py         # DeltaNetDecoderBlock (mlp_class parameter)
  qwen35_utils.py           # Weight conversion (MoE key protection)
  model_config.py           # Config parsing (MoE fields)

models/demos/qwen35/
  demo/demo.py              # 27B demo
  demo/demo_a3b.py          # A3B demo
  tests/test_pcc.py         # 27B PCC tests
  tests/test_a3b_pcc.py     # A3B PCC tests

ttnn/.../gated_delta_net/device/
  kernels/gated_delta_net_compute.cpp     # Fused kernel
  gated_delta_net_program_factory.cpp     # CB layout
```

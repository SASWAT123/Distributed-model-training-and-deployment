# Module 04: ZeRO Optimization

**Time**: ~60 minutes
**Goal**: Understand what ZeRO shards and why. Implement ZeRO-3 using PyTorch FSDP. Run DeepSpeed ZeRO for comparison.

---

## The Problem with DDP's Memory Usage

In standard DDP, every GPU holds a complete copy of everything:

```
GPU 0 holds:
  - Model parameters (fp16):    2 bytes × N params
  - Gradients (fp16):           2 bytes × N params
  - Optimizer state — m (fp32): 4 bytes × N params
  - Optimizer state — v (fp32): 4 bytes × N params
  - Master weights (fp32):      4 bytes × N params
  ─────────────────────────────────────────────────
  Total:                       16 bytes × N params

GPU 1 holds: identical copy
GPU 2 holds: identical copy
GPU 3 holds: identical copy
```

With 4 GPUs, you are storing 4 copies of every optimizer state. That is 3 copies of pure redundancy.

**ZeRO (Zero Redundancy Optimizer)** eliminates this redundancy by sharding state across GPUs.

---

## The Three ZeRO Stages

### ZeRO Stage 1: Shard Optimizer States

Each GPU holds the full model and all gradients, but only 1/N of the optimizer states.

```
GPU 0: [params][grads][opt_states: shard 0]
GPU 1: [params][grads][opt_states: shard 1]

Memory per GPU: 2+2+16/N bytes per param
```

Speedup: same as DDP. Memory reduction: ~4x with 4 GPUs.

### ZeRO Stage 2: Shard Gradients + Optimizer States

Each GPU holds the full model but only 1/N of gradients and optimizer states.

```
GPU 0: [params][grads: shard 0][opt_states: shard 0]
GPU 1: [params][grads: shard 1][opt_states: shard 1]

Memory per GPU: 2 + (2+16)/N bytes per param
```

Memory reduction: ~8x with 4 GPUs. Small communication overhead.

### ZeRO Stage 3: Shard Everything (Parameters + Gradients + Optimizer States)

Each GPU only holds 1/N of everything.

```
GPU 0: [params: shard 0][grads: shard 0][opt_states: shard 0]
GPU 1: [params: shard 1][grads: shard 1][opt_states: shard 1]

Memory per GPU: (2+2+16)/N bytes per param
```

Memory reduction: ~16x with 4 GPUs. More communication (parameters must be gathered during forward/backward).

---

## The Communication Overhead

Nothing is free. ZeRO-3 requires gathering parameter shards from all GPUs during every forward and backward pass. This adds AllGather and ReduceScatter operations:

```
Forward pass with ZeRO-3:
  for each layer:
    AllGather(param shard 0..N)  ← gather full layer params
    compute forward              ← normal computation
    discard params               ← free memory
```

With fast NVLink interconnects (e.g., A100s), this overhead is small (~5-10%). On PCIe (T4s), it is larger. On a real cluster with InfiniBand, ZeRO-3 is often faster than DDP for large models.

---

## PyTorch FSDP vs DeepSpeed

Both implement ZeRO-3 but with different APIs:

| | PyTorch FSDP | DeepSpeed ZeRO |
|--|--|--|
| Integrated with | PyTorch native | External library |
| Config | Python API | JSON config file |
| CPU offload | Yes | Yes (more mature) |
| NVMe offload | No | Yes (ZeRO-Infinity) |
| Mixed precision | Via `MixedPrecision` policy | Via `fp16` config |
| Best for | Standard PyTorch workflows | Production, very large models |

This module covers FSDP. The notebook also includes a DeepSpeed example.

---

## Sharding Strategies in FSDP

```python
from torch.distributed.fsdp import ShardingStrategy

ShardingStrategy.FULL_SHARD      # ZeRO-3: shard params + grads + optimizer
ShardingStrategy.SHARD_GRAD_OP   # ZeRO-2: shard grads + optimizer only
ShardingStrategy.NO_SHARD        # DDP (no sharding — for comparison)
ShardingStrategy.HYBRID_SHARD    # ZeRO-3 within a node, replicate across nodes
```

---

## The Auto-Wrap Policy

FSDP needs to know which modules to shard independently. You use a "wrap policy":

```python
from torch.distributed.fsdp.wrap import size_based_auto_wrap_policy

# Wrap any module with more than 100M parameters
policy = functools.partial(size_based_auto_wrap_policy, min_num_params=100_000_000)
```

Wrapping too finely = lots of communication overhead.
Wrapping too coarsely = large peak memory spikes during AllGather.
For transformers, wrapping at the transformer layer level is usually optimal.

---

## CPU Offloading

FSDP can offload parameters to CPU RAM when not in use:

```python
from torch.distributed.fsdp.fully_sharded_data_parallel import CPUOffload

model = FSDP(model, cpu_offload=CPUOffload(offload_params=True))
```

This lets you train models much larger than total GPU VRAM. The cost: slower training (CPU-GPU transfers become the bottleneck).

---

## Exercise

1. **Exercise A**: Train with `ShardingStrategy.NO_SHARD` (baseline DDP) and record memory
2. **Exercise B**: Switch to `FULL_SHARD` and observe memory reduction
3. **Exercise C**: Enable CPU offload and try training a model that would normally OOM
4. **Exercise D**: Run DeepSpeed ZeRO-2 and compare to FSDP

---

## Next: Open `notebook.ipynb`

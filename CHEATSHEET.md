# Distributed Training Cheat Sheet

Quick reference for concepts and commands from this tutorial.

---

## When to Use What

| Problem | Solution | Tool |
|---------|----------|------|
| Training too slow | Data Parallelism | DDP or Accelerate |
| Model too big for 1 GPU | Model / Pipeline Parallelism | Manual or Pipe |
| Too much optimizer state | ZeRO Stage 1 | FSDP SHARD_GRAD_OP or DeepSpeed ZeRO-1 |
| Too many gradients | ZeRO Stage 2 | FSDP SHARD_GRAD_OP or DeepSpeed ZeRO-2 |
| Everything too big | ZeRO Stage 3 | FSDP FULL_SHARD or DeepSpeed ZeRO-3 |
| Model + slow | DDP + large model | FSDP FULL_SHARD |
| I don't want to think about it | Just use Accelerate | Accelerate |

---

## Memory Per Parameter

| What is stored | Bytes per param | Notes |
|---------------|-----------------|-------|
| fp16 params | 2 | Inference |
| fp32 params | 4 | |
| fp16 grads | 2 | |
| Adam m (fp32) | 4 | Optimizer state |
| Adam v (fp32) | 4 | Optimizer state |
| fp32 master copy | 4 | For stable fp16 training |
| **Total (mixed precision training)** | **16** | |

---

## Key Commands

```bash
# Run DDP training (2 GPUs)
torchrun --nproc_per_node=2 train.py

# Run with Accelerate config
accelerate launch --config_file config.yaml train.py

# Run with DeepSpeed
deepspeed --num_gpus=2 train.py --deepspeed ds_config.json

# Generate Accelerate config interactively
accelerate config

# Check GPU status
nvidia-smi
watch -n 1 nvidia-smi  # refresh every second
```

---

## Boilerplate: DDP

```python
import os, torch, torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

def main():
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    dist.init_process_group(backend="nccl")
    torch.cuda.set_device(local_rank)

    model = MyModel().to(local_rank)
    model = DDP(model, device_ids=[local_rank])

    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank)
    loader = DataLoader(dataset, batch_size=32, sampler=sampler)

    for epoch in range(n_epochs):
        sampler.set_epoch(epoch)   # ← don't forget this
        for x, y in loader:
            x, y = x.to(local_rank), y.to(local_rank)
            loss = criterion(model(x), y)
            loss.backward()       # ← AllReduce fires here
            optimizer.step()

    dist.destroy_process_group()

if __name__ == "__main__":
    main()
```

---

## Boilerplate: FSDP

```python
import functools
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP, ShardingStrategy
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy

wrap_policy = functools.partial(
    transformer_auto_wrap_policy,
    transformer_layer_cls={nn.TransformerEncoderLayer}
)

model = FSDP(
    model,
    auto_wrap_policy=wrap_policy,
    sharding_strategy=ShardingStrategy.FULL_SHARD,   # ZeRO-3
    device_id=local_rank,
)
```

---

## Boilerplate: Accelerate

```python
from accelerate import Accelerator

accelerator = Accelerator()  # reads config from env

model, optimizer, loader = accelerator.prepare(model, optimizer, loader)

for x, y in loader:
    optimizer.zero_grad()
    loss = criterion(model(x), y)
    accelerator.backward(loss)  # replaces loss.backward()
    optimizer.step()

# Save
accelerator.wait_for_everyone()
accelerator.save(accelerator.unwrap_model(model).state_dict(), "ckpt.pt")
```

---

## Common Bugs

| Error | Cause | Fix |
|-------|-------|-----|
| `Expected tensors on same device` | Input not moved to rank | `.to(rank)` or `.to(local_rank)` |
| `Address already in use` | Port occupied by crashed run | Change `MASTER_PORT` |
| Gradients are None | Unused parameters | `DDP(..., find_unused_parameters=True)` |
| Loss identical every epoch | Missing `set_epoch()` | `sampler.set_epoch(epoch)` |
| `RuntimeError: RRef` | Forgot `.local_value()` on Pipe output | `out = out_rref.local_value()` |
| OOM with FSDP save | Default state dict is sharded | Use `StateDictType.FULL_STATE_DICT` |
| Duplicate logging | Not gating on rank 0 | `if rank == 0: print(...)` or `accelerator.print(...)` |

---

## ZeRO Stage Summary

| Stage | What is sharded | Memory reduction | Communication overhead |
|-------|----------------|-----------------|----------------------|
| DDP | Nothing | 1x | AllReduce grads |
| ZeRO-1 | Optimizer states | ~4x | AllReduce grads |
| ZeRO-2 | Optimizer + grads | ~8x | ReduceScatter grads |
| ZeRO-3 | Optimizer + grads + params | ~16x | AllGather params + ReduceScatter grads |

---

## Gradient Checkpointing (Memory vs Compute)

```python
from torch.utils.checkpoint import checkpoint

# Instead of: x = layer(x)
# Use:
x = checkpoint(layer, x, use_reentrant=False)

# Or enable globally on HuggingFace models:
model.gradient_checkpointing_enable()
```

Memory: ~50% reduction in activations
Compute: ~33% increase (recomputes activations during backward)

---

## Mixed Precision

```python
# Manual
with torch.autocast(device_type='cuda', dtype=torch.float16):
    output = model(input)

# Via Accelerate
accelerator = Accelerator(mixed_precision="fp16")
# (or bf16 for A100/H100/RTX30xx, better numerics than fp16)
```

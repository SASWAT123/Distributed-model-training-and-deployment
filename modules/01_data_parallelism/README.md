# Module 01: Data Parallelism (DDP)

**Time**: ~60 minutes
**Goal**: Train a model across both GPUs using PyTorch's DistributedDataParallel. Understand gradient synchronization. Beat your Module 00 baseline by ~1.8x.

---

## The Concept

Data parallelism is the most common form of distributed training. The idea is simple:

1. Put a **full copy of the model** on every GPU
2. Split each batch — each GPU processes a different chunk
3. After the backward pass, **average the gradients** across all GPUs
4. Every GPU now has identical gradients, so all model copies stay in sync

```
                    Batch: 64 samples
                          ↓
           ┌──────────────┴──────────────┐
           │                             │
    GPU 0: 32 samples              GPU 1: 32 samples
    Full model copy                Full model copy
           │                             │
    forward pass                   forward pass
    loss = 0.8                     loss = 0.7
    backward pass                  backward pass
    local gradients                local gradients
           │                             │
           └──────── AllReduce ──────────┘
                  avg(grads) synced
                  to both GPUs
           │                             │
    optimizer.step()            optimizer.step()
    (identical on both)         (identical on both)
```

The `AllReduce` operation — averaging gradients across all GPUs — is the communication cost of DDP. Everything else is pure computation.

---

## Why DDP Instead of DataParallel?

PyTorch has two data parallelism APIs:

| API | How it works | Problem |
|-----|-------------|---------|
| `nn.DataParallel` | One process, multiple threads. GPU 0 gathers all results. | GPU 0 bottleneck. Thread contention. Slow. |
| `nn.DistributedDataParallel` | One process per GPU. Communication via NCCL. | None for single-node. Use this always. |

**Always use DDP. Never use DataParallel for new code.**

---

## How DDP Works Under the Hood

When you call `model = DDP(model)`, PyTorch registers a hook on every parameter's `.grad`. After each `.backward()` call, as soon as a gradient is ready, it immediately fires an **AllReduce** to sync that gradient across all processes. This happens **while the rest of backward is still running** — communication and computation overlap.

```
backward pass timeline:
  compute grad(layer_N)  →  AllReduce(grad_N)  ─────────────→  done
  compute grad(layer_N-1) →  AllReduce(grad_N-1) ──────────→  done
  ...                                                         (all overlap)
```

This is called **gradient bucketing** and it's why DDP is fast.

---

## The DistributedSampler

Without a sampler, every GPU would load the same data (same shuffle seed). Each GPU would waste compute doing identical work.

`DistributedSampler` partitions the dataset so each GPU sees a non-overlapping subset:

```python
# GPU 0 processes: samples [0, 2, 4, 6, ...]
# GPU 1 processes: samples [1, 3, 5, 7, ...]

sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank)
loader = DataLoader(dataset, batch_size=32, sampler=sampler)
```

**Important**: Call `sampler.set_epoch(epoch)` at the start of each epoch. Without this, the shuffle is the same every epoch.

---

## The Launch Command

DDP requires launching multiple processes. You do this with `torchrun`:

```bash
torchrun --nproc_per_node=2 train.py
```

`--nproc_per_node=2` means: spawn 2 processes on this machine, one per GPU.

`torchrun` automatically sets these environment variables for each process:
- `RANK` — this process's global rank
- `LOCAL_RANK` — this process's rank on this machine (same as RANK for single-node)
- `WORLD_SIZE` — total number of processes

Your training script reads these to initialize the process group.

---

## Exercise

In the notebook, you will:

1. **Exercise A**: Run the complete DDP training script and observe that both GPUs are used
2. **Exercise B**: Deliberately break it (remove `sampler.set_epoch(epoch)`) and observe the duplicate data problem
3. **Exercise C**: Profile and compare throughput against your Module 00 baseline
4. **Exercise D**: Scale the batch size with world size and observe the effect on loss curves

---

## Common Bugs

**"Expected all tensors to be on the same device"**
→ You forgot to move data to `rank`. Add `.to(rank)` to your inputs.

**"Address already in use"**
→ A previous training run crashed and left the port open. Change `MASTER_PORT` to a different number.

**Gradient is None on some params**
→ You have unused parameters. Add `find_unused_parameters=True` to DDP constructor.

**Only GPU 0 is being used**
→ You are using `DataParallel` instead of `DDP`. Or you forgot to call `mp.spawn`.

---

## Next: Open `notebook.ipynb`

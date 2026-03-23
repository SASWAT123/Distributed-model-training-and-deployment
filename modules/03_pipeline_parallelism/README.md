# Module 03: Pipeline Parallelism

**Time**: ~45 minutes
**Goal**: Fix the GPU idle time from Module 02 using micro-batches. Understand the pipeline schedule.

---

## The Insight

In naive model parallelism, GPUs take turns being active. The fix is to process multiple mini-batches at the same time — while GPU 1 is processing mini-batch 1, GPU 0 starts processing mini-batch 2.

This is exactly how a CPU instruction pipeline works. Instead of finishing one instruction before fetching the next, you overlap fetch, decode, and execute stages.

---

## The Pipeline Schedule

Split your batch into smaller **micro-batches**. Feed them through the pipeline one at a time:

```
              Time →
Micro-batch:  mb1    mb2    mb3    mb4

GPU 0 (Stage 0):
  [mb1 fwd]
             [mb2 fwd]
                    [mb3 fwd]
                           [mb4 fwd]
                                  [mb4 bwd]
                                         [mb3 bwd]
                                                [mb2 bwd]
                                                       [mb1 bwd]

GPU 1 (Stage 1):
         [mb1 fwd]
                [mb2 fwd]
                       [mb3 fwd]
                              [mb4 fwd]
                                     [mb4 bwd]
                                            [mb3 bwd]
                                                   [mb2 bwd]
                                                          [mb1 bwd]
```

The "pipeline bubble" (idle time at start and end) shrinks as `num_micro_batches` grows. With 8 micro-batches, the bubble is only 1/8 of total time.

---

## The Tradeoff

More micro-batches = smaller bubble = better GPU utilization, BUT:

- More micro-batches means more activations in memory simultaneously
- Too many micro-batches can OOM

The optimal number of micro-batches is typically 4-8 for a 2-stage pipeline.

---

## GPipe vs. 1F1B

There are two main pipeline schedules:

**GPipe** (what PyTorch's `Pipe` uses by default):
- Run all micro-batch forwards, then all micro-batch backwards
- Simple but memory-hungry (stores all activations)

**1F1B (One Forward One Backward)** (used by Megatron-LM):
- Alternates forward and backward passes
- Same pipeline efficiency but lower peak memory
- More complex to implement

For 2 GPUs on Kaggle, GPipe is fine. 1F1B matters when you have many pipeline stages.

---

## Gradient Checkpointing

Pipeline parallelism stores activations for all in-flight micro-batches. For long sequences, this can be a lot of memory. **Gradient checkpointing** trades memory for compute: instead of saving activations, recompute them during backward pass.

```python
from torch.utils.checkpoint import checkpoint

# Instead of:
x = layer(x)  # saves activation

# Use:
x = checkpoint(layer, x)  # recomputes activation during backward
```

This roughly halves activation memory at the cost of ~33% more compute.

---

## PyTorch's Pipe API

PyTorch provides a `torch.distributed.pipeline.sync.Pipe` class:

```python
from torch.distributed.pipeline.sync import Pipe

model = Pipe(
    nn.Sequential(stage0_on_gpu0, stage1_on_gpu1),
    chunks=4  # Number of micro-batches
)

# Returns an RRef (remote reference), not a tensor
output_rref = model(input)
output = output_rref.local_value()  # Get the actual tensor
```

There are limitations to this API — the notebook covers the gotchas.

---

## Exercise

In the notebook you will:

1. **Exercise A**: Run the pipeline parallel model and compare GPU utilization to Module 02
2. **Exercise B**: Vary `chunks` (micro-batches) from 1 to 8 and plot throughput vs. memory
3. **Exercise C**: Add gradient checkpointing and measure the memory savings
4. **Exercise D**: Implement a manual pipeline with `asyncio`-style overlap (advanced)

---

## Next: Open `notebook.ipynb`

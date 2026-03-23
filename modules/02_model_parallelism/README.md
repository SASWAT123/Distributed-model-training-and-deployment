# Module 02: Model Parallelism

**Time**: ~45 minutes
**Goal**: Split a model that doesn't fit on one GPU across two GPUs. Understand the GPU idle problem and why it motivates pipeline parallelism.

---

## The Problem DDP Cannot Solve

DDP requires each GPU to hold a **full copy of the model**. If your model is 20GB and each GPU has 16GB, DDP fails.

```
Model size: 20GB
GPU 0: 16GB  ← model doesn't fit
GPU 1: 16GB  ← model doesn't fit
```

Model parallelism solves this by putting **different parts of the model** on different GPUs.

---

## How Model Parallelism Works

```
Input (on CPU or GPU 0)
        ↓
  ┌─────────────┐
  │   GPU 0     │  ← Layers 0..N/2 live here
  │  Embedding  │
  │  Layer 0    │
  │  Layer 1    │
  │  Layer 2    │
  └──────┬──────┘
         │  tensor moves across GPUs (the bottleneck)
  ┌──────▼──────┐
  │   GPU 1     │  ← Layers N/2..N live here
  │  Layer 3    │
  │  Layer 4    │
  │  Layer 5    │
  │ Classifier  │
  └─────────────┘
        ↓
    Output (on GPU 1)
```

The key: when the forward pass hits the GPU boundary, the activations are copied from GPU 0 to GPU 1. During backward, gradients flow the other direction.

---

## The Hidden Problem: GPU Idle Time

Here is what GPU utilization actually looks like during model parallel training:

```
Time →
GPU 0: [FORWARD][idle ][idle ][BACKWARD][ idle  ]
GPU 1: [idle   ][FORWARD][BACKWARD][idle      ]

Most of the time, one GPU is doing nothing.
```

This is called the **bubble problem**. In naive model parallelism, only one GPU is active at a time. Your 2 GPUs run at approximately 1 GPU speed — no speedup, just more memory.

This is why **pipeline parallelism** (Module 03) exists: it fills the bubble with micro-batches.

---

## When to Use Model Parallelism

Use model parallelism (or its relatives) when:
- Your model is too large for a single GPU
- You want to fit a larger model than VRAM allows
- You are training very large models (>10B parameters)

Do NOT use it as a speed optimization — it won't help with that. DDP is for speed. Model parallelism is for memory.

---

## Tensor Parallelism vs. Layer Parallelism

This module covers **layer parallelism** (different layers on different GPUs).

There is a more advanced variant called **tensor parallelism** where you split individual layer weight matrices across GPUs. This is what Megatron-LM uses for GPT-scale models:

```
Layer Parallelism:
  GPU 0: [Attention Layer 0]
  GPU 1: [Attention Layer 1]

Tensor Parallelism (within one layer):
  GPU 0: [Left half of Attention Weight Matrix]
  GPU 1: [Right half of Attention Weight Matrix]
```

This module covers layer parallelism. Tensor parallelism requires specialized code (Megatron-LM).

---

## Exercise

In the notebook you will:

1. **Exercise A**: Build a model that causes OOM on a single GPU, then fix it with model parallelism
2. **Exercise B**: Profile GPU utilization and see the bubble problem with your own eyes
3. **Exercise C**: Measure how much memory each GPU uses vs. single-GPU baseline
4. **Exercise D**: Implement a manual `forward` that moves tensors across GPUs correctly

---

## Next: Open `notebook.ipynb`

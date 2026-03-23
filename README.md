# Distributed Model Training on Kaggle Free GPUs
### A hands-on tutorial for learning data parallelism, model parallelism, pipeline parallelism, and ZeRO optimizations

---

## What You Will Learn

By the end of this tutorial you will be able to:

- Explain the difference between data parallelism, model parallelism, and pipeline parallelism
- Write training loops using PyTorch DDP, FSDP, and DeepSpeed
- Understand what ZeRO stages 1/2/3 actually shard and why it matters
- Profile GPU memory and throughput to compare strategies
- Use HuggingFace Accelerate to switch strategies with one line of config

---

## Prerequisites

- Basic Python and PyTorch (you know what a DataLoader and an optimizer are)
- A Kaggle account (free at kaggle.com)
- No prior distributed training experience needed

---

## How to Use This Tutorial

Each module is a self-contained Kaggle notebook. Work through them in order — each one builds on the last.

```
modules/
├── 00_setup_and_profiling/     ← Start here. Verify your environment.
├── 01_data_parallelism/        ← DDP. The most important module.
├── 02_model_parallelism/       ← Split a model across GPUs.
├── 03_pipeline_parallelism/    ← Keep GPUs busy with micro-batches.
├── 04_zero_optimization/       ← FSDP and DeepSpeed ZeRO.
└── 05_accelerate/              ← Tie everything together cleanly.
```

Each module folder contains:
- `notebook.ipynb` — the Kaggle notebook to run
- `README.md` — concept explanation before you touch any code
- `solution/` — reference implementation if you get stuck

---

## Kaggle Setup (Do This Once)

1. Go to [kaggle.com](https://kaggle.com) and sign in
2. Click **Create** → **New Notebook**
3. On the right sidebar: **Settings** → **Accelerator** → select **GPU T4 x2**
4. On the right sidebar: **Settings** → **Internet** → turn **On**
5. You now have 2x NVIDIA T4 GPUs (16GB each) and ~30 hrs/week free

> **Tip**: Kaggle sessions time out after 12 hours of inactivity. Save your notebook often with Ctrl+S.

---

## Your Hardware

```
┌─────────────────────────────────────────┐
│           Kaggle Notebook Session       │
│                                         │
│   ┌──────────────┐   ┌──────────────┐   │
│   │    GPU 0     │   │    GPU 1     │   │
│   │  T4  16 GB   │   │  T4  16 GB   │   │
│   │  VRAM        │   │  VRAM        │   │
│   └──────┬───────┘   └───────┬──────┘   │
│          └────── PCIe ───────┘          │
│                                         │
│   RAM: ~29 GB    CPU: 4 cores           │
└─────────────────────────────────────────┘
```

This is your "distributed cluster." 2 GPUs is enough to learn every strategy in this tutorial. Everything you learn here transfers directly to real multi-node clusters.

---

## The Core Problem (Read This First)

Distributed training exists because of two separate problems:

```
┌─────────────────────────────────────────────────────────┐
│  PROBLEM 1: Training is too slow                        │
│                                                         │
│  Solution: Split the DATA across GPUs                   │
│  Name: Data Parallelism                                 │
│  Tools: DDP, Accelerate                                 │
├─────────────────────────────────────────────────────────┤
│  PROBLEM 2: Model is too big to fit in one GPU          │
│                                                         │
│  Solution A: Split the MODEL LAYERS across GPUs         │
│  Name: Model Parallelism / Tensor Parallelism           │
│                                                         │
│  Solution B: Split the MODEL into TIME STAGES           │
│  Name: Pipeline Parallelism                             │
│                                                         │
│  Solution C: Shard optimizer state, grads, and params   │
│  Name: ZeRO (Zero Redundancy Optimizer)                 │
│  Tools: FSDP, DeepSpeed                                 │
├─────────────────────────────────────────────────────────┤
│  PROBLEM 3: Both slow AND too big                       │
│                                                         │
│  Solution: Combine the above (3D parallelism)           │
│  Used by: GPT-4, LLaMA training runs                    │
└─────────────────────────────────────────────────────────┘
```

---

## Start with Module 00 →

[modules/00_setup_and_profiling/README.md](modules/00_setup_and_profiling/README.md)

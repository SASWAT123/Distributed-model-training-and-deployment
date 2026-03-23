# Module 00: Setup and Profiling

**Time**: ~30 minutes
**Goal**: Verify your Kaggle environment, install dependencies, and build a profiling utility you will use in every subsequent module.

---

## Before You Write Any Code

Run the notebook for this module (`notebook.ipynb`) cell by cell. At the end you should see:

```
PyTorch version: 2.x.x
CUDA available: True
Number of GPUs: 2
  GPU 0: Tesla T4, 16.0 GB
  GPU 1: Tesla T4, 16.0 GB
All checks passed. You are ready to start.
```

If you only see 1 GPU, go back to Kaggle settings and make sure you selected **GPU T4 x2**, not just GPU T4.

---

## Concept: What Is a Process Group?

Distributed training in PyTorch works by launching **multiple processes** — one per GPU. Each process thinks it is running a normal training script. The magic is that they communicate behind the scenes.

```
torchrun --nproc_per_node=2 train.py
         ↓
  spawns 2 processes:

  Process 0 (rank=0):          Process 1 (rank=1):
  - controls GPU 0             - controls GPU 1
  - runs train.py              - runs train.py
  - rank=0, world_size=2       - rank=1, world_size=2
         ↕ communicate via NCCL ↕
```

Three terms you must know:

| Term | Meaning |
|------|---------|
| `rank` | This process's ID. rank=0 is always the "main" process. |
| `world_size` | Total number of processes (= number of GPUs in most cases). |
| `NCCL` | NVIDIA's communication library. Handles GPU-to-GPU data transfer. |

---

## Concept: The Profiler You Will Use

Every module uses the same `TrainingProfiler` class. It reports:

- **Step time** — how long each training step takes
- **Throughput** — samples processed per second (the key metric)
- **GPU memory** — how much VRAM each GPU is using
- **GPU utilization** — what % of GPU compute is active

When you compare strategies, these four numbers tell the whole story.

---

## Now Open the Notebook

Open `notebook.ipynb` and run it. Come back here when you are done.

---

## Checkpoint Questions

Before moving to Module 01, you should be able to answer:

1. What does `rank` mean? What is `world_size`?
2. Why do we need NCCL?
3. On your Kaggle instance, what is the total available VRAM across both GPUs?
4. What command do we use to launch a multi-process training job?

Answers are at the bottom of the notebook.

---

## Next: [Module 01 - Data Parallelism](../01_data_parallelism/README.md)

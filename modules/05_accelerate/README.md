# Module 05: HuggingFace Accelerate

**Time**: ~45 minutes
**Goal**: Use Accelerate to write one training script that can run under DDP, FSDP, or DeepSpeed by changing a single config file. Fine-tune a real pretrained model.

---

## Why Accelerate?

After the previous four modules, you have written a lot of boilerplate:

- `dist.init_process_group()`
- `DistributedSampler`
- `model = DDP(model, device_ids=[rank])`
- `sampler.set_epoch(epoch)`
- `accelerator.backward(loss)` instead of `loss.backward()`
- Different save/load logic for FSDP vs DDP

Accelerate hides all of this. Your training script looks almost identical to single-GPU code:

```python
# Before Accelerate:
dist.init_process_group('nccl')
model = DDP(model.to(rank), device_ids=[rank])
sampler = DistributedSampler(dataset, ...)
loader = DataLoader(dataset, sampler=sampler)
x = x.to(rank)
loss.backward()
optimizer.step()

# After Accelerate:
accelerator = Accelerator()
model, optimizer, loader = accelerator.prepare(model, optimizer, loader)
# x stays on CPU — accelerator moves it automatically
accelerator.backward(loss)
optimizer.step()
```

The `accelerator.prepare()` call detects your environment and applies the right wrapping strategy.

---

## Switching Strategies with Config

The power of Accelerate is that you switch between DDP, FSDP, and DeepSpeed **without changing your Python code**. You just change a config file:

```bash
# Generate a config interactively
accelerate config

# Or write one programmatically
accelerate config default
```

```yaml
# accelerate_config_ddp.yaml
compute_environment: LOCAL_MACHINE
distributed_type: MULTI_GPU
num_processes: 2

# accelerate_config_fsdp.yaml
compute_environment: LOCAL_MACHINE
distributed_type: FSDP
fsdp_config:
  fsdp_sharding_strategy: FULL_SHARD
num_processes: 2

# accelerate_config_deepspeed.yaml
compute_environment: LOCAL_MACHINE
distributed_type: DEEPSPEED
deepspeed_config:
  zero_stage: 3
num_processes: 2
```

Then launch:
```bash
accelerate launch --config_file accelerate_config_fsdp.yaml train.py
```

---

## Mixed Precision

Mixed precision (fp16 or bf16) is a free ~2x memory and speed improvement. Enable it with:

```python
accelerator = Accelerator(mixed_precision="fp16")
# or bf16 if your GPU supports it (A100, H100, RTX 30xx+)
# T4 supports fp16 but not bf16
```

Or in config:
```yaml
mixed_precision: fp16
```

Under the hood, Accelerate uses `torch.autocast` for forward passes and `GradScaler` to avoid fp16 underflow during backward.

---

## Practical: Fine-Tuning GPT-2

In this module you will fine-tune GPT-2 small (~117M params) on a text classification task using Accelerate. This is as close to "real work" as you can get on free Kaggle GPUs.

GPT-2 small fits easily on one T4. We use 2 GPUs to practice the workflow, not because we need to.

---

## Exercise

1. **Exercise A**: Run the Accelerate training script with DDP config. Record throughput.
2. **Exercise B**: Switch to FSDP config (no code changes). Record throughput and memory.
3. **Exercise C**: Enable `mixed_precision="fp16"` and re-run. Compare memory.
4. **Exercise D**: Add `gradient_checkpointing` to the model and observe memory drop.

---

## Next Steps After This Tutorial

Once you finish all 5 modules, you have solid foundations. Next topics to explore:

- **Tensor Parallelism**: Megatron-LM splits individual weight matrices. Relevant for 70B+ models.
- **3D Parallelism**: GPT-4 training combined data + pipeline + tensor parallelism.
- **Flash Attention**: Rewrite the attention computation to be IO-bound instead of memory-bound.
- **`torch.compile`**: PyTorch 2.0 graph compilation, often a free 20-30% speedup.
- **Multi-node training**: Connect multiple machines via NCCL over InfiniBand.

---

## Open `notebook.ipynb`

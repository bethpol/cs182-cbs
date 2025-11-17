# Branching Experiments: Testing & Gradient Accumulation Guide

This document covers:
1. **How to run tests** for the branching experiment code
2. **How gradient accumulation works** and why it's needed
3. **How to configure it** for your hardware

---

## Running Tests

### Prerequisites

```bash
pip install pytest
```

### Running All Tests

```bash
# Run all tests
pytest test_*.py -v

# Run specific test file
pytest test_generate_branch_configs.py -v
pytest test_run_branch_sweep.py -v
pytest test_auto_gradient_accumulation.py -v

# Run with coverage
pytest test_*.py -v --cov=. --cov-report=html
```

### What's Being Tested

1. **`test_generate_branch_configs.py`** (114 tests)
   - Config file generation
   - Scaling formulas (batch size, learning rate)
   - Parameter validation
   - Edge cases

2. **`test_run_branch_sweep.py`** (25 tests)
   - Branch execution
   - Results collection
   - Error handling
   - Log file management

3. **`test_auto_gradient_accumulation.py`** (40+ tests)
   - Gradient accumulation calculations
   - Memory estimation
   - Auto-configuration pipeline
   - Realistic scenarios

---

## Understanding Gradient Accumulation

### The Problem

When you scale batch size by factor `k`, the batch may not fit in GPU memory:

```
k=1:  batch_size = 8   ✓ Fits in GPU
k=2:  batch_size = 16  ✓ Fits in GPU  
k=4:  batch_size = 32  ✓ Fits in GPU
k=8:  batch_size = 64  ⚠️  Might not fit
k=16: batch_size = 128 ❌ Doesn't fit
k=32: batch_size = 256 ❌ Doesn't fit
```

### The Solution

**Gradient accumulation** splits a large batch into smaller "micro-batches":

```
Effective Batch = Micro-Batch × Gradient Accumulation Steps

Example for k=16 (target batch size = 128):
  Micro-batch size: 16 (fits in GPU)
  Grad accum steps: 8
  Effective batch:  16 × 8 = 128 ✓
```

### How It Works

```python
for step in range(max_steps):
    optimizer.zero_grad()
    
    # Accumulate gradients over multiple micro-batches
    for micro_batch in split_into_micro_batches(batch):
        loss = model(micro_batch)
        loss.backward()  # Accumulate gradients
    
    # Update only after all micro-batches
    optimizer.step()
```

**Key insight:** The model updates are mathematically identical to using the full batch!

---

## Automatic Configuration

### Quick Start

The **automatic gradient accumulation** calculates the optimal micro-batch size and accumulation steps:

```python
from auto_gradient_accumulation import auto_configure_batch_size

config = auto_configure_batch_size(
    target_batch_size=256,      # What we want (k × base_batch)
    model_params=45_000_000,    # Your model size
    sequence_length=128,        # Sequence/context length
    gpu_memory_gb=24           # Your GPU (None = auto-detect)
)

print(config)
# Output: Micro-batch: 32, Grad accum: 8, Effective: 256
```

### Strategy Options

#### Option 1: Manual (Recommended)

If you know what fits on your GPU:

```python
# In generate_branch_configs_with_grad_accum.py
MAX_MICRO_BATCH_SIZE = 32  # What fits on your GPU
```

**How to find this:**
- Start with your base batch size
- Double it until OOM (Out of Memory)
- Use the largest that works

#### Option 2: Automatic Estimation

Let the code estimate:

```python
MAX_MICRO_BATCH_SIZE = None  # Enable auto-estimation
MODEL_PARAMS = 45_000_000
GPU_MEMORY_GB = None  # Auto-detect or specify
```

The estimation uses this formula:
```
available_memory = gpu_memory × 0.7 (safety margin)
model_memory = params × 16 bytes (mixed precision + optimizer)
activation_memory = available - model
max_batch = activation_memory / (params × seq_len × 4)
```

---

## Testing Gradient Accumulation on Your Hardware

### Step 1: Check What Fits

Run the test script:

```bash
python auto_gradient_accumulation.py
```

This prints:
- Estimated max batch sizes for different model/GPU configs
- Configuration tables for your k-values
- Memory calculations

### Step 2: Empirical Testing

Create a small test script to find your actual limit:

```python
# test_max_batch.py
import torch
from your_model import Model

model = Model().cuda()
optimizer = torch.optim.AdamW(model.parameters())

# Test increasing batch sizes
for batch_size in [8, 16, 32, 64, 128]:
    try:
        print(f"Testing batch_size={batch_size}...")
        x = torch.randint(0, 50257, (batch_size, 128)).cuda()
        
        optimizer.zero_grad()
        loss = model(x)
        loss.backward()
        optimizer.step()
        
        print(f"  ✓ batch_size={batch_size} works!")
        
    except RuntimeError as e:
        if "out of memory" in str(e):
            print(f"  ✗ batch_size={batch_size} OOM!")
            break
        raise
```

### Step 3: Configure

Update `generate_branch_configs_with_grad_accum.py`:

```python
MAX_MICRO_BATCH_SIZE = 32  # Use the max that worked in Step 2
```

### Step 4: Generate Configs

```bash
python generate_branch_configs_with_grad_accum.py
```

This creates configs with automatic gradient accumulation:

```
Batch Size Configuration Table
===============================================================================
k    | Target Batch | Micro-Batch  | Grad Accum  | Effective 
-------------------------------------------------------------------------------
1    | 8            | 8            | 1           | 8         
2    | 16           | 16           | 1           | 16        
4    | 32           | 32           | 1           | 32        
8    | 64           | 32           | 2           | 64        
16   | 128          | 32           | 4           | 128       
32   | 256          | 32           | 8           | 256       
===============================================================================
```

---

## Comparing Config Generators

### Original Version
```bash
python generate_branch_configs.py
```
- ✓ Simple, straightforward
- ✗ All configs use `gradient_accumulation_steps = 1`
- ✗ Large k-values will OOM

### Enhanced Version
```bash
python generate_branch_configs_with_grad_accum.py
```
- ✓ Automatic gradient accumulation
- ✓ Memory-aware configuration
- ✓ Large k-values work
- ✓ Shows configuration table

**Recommendation:** Use the enhanced version unless you know all batches fit.

---

## Best Practices

### For Gradient Accumulation

1. **Divisibility Matters**
   ```python
   # Good: 128 = 32 × 4 (exact)
   effective_batch = 128
   micro_batch = 32
   grad_accum = 4
   
   # Acceptable: 130 ≈ 32 × 4 (close enough)
   # The slight difference won't affect convergence
   ```

2. **Logging Frequency**
   - With gradient accumulation, you do fewer optimizer steps
   - Adjust logging intervals accordingly:
   ```python
   # Log by tokens (not steps) to stay consistent
   log_interval_tokens = 10_000  # Same across all k-values
   ```

3. **Reproducibility**
   - Same effective batch size → same training dynamics
   - Gradient accumulation is mathematically equivalent
   - Results should match regardless of micro-batch size

### For Testing

1. **Start with small k-values**
   - Test k=1, 2, 4 first
   - These should work without gradient accumulation

2. **Monitor GPU memory**
   ```bash
   watch -n 1 nvidia-smi
   ```

3. **Check effective batch sizes**
   - Verify: `micro_batch × grad_accum = k × base_batch`

---

## Troubleshooting

### OOM Even With Gradient Accumulation

1. **Reduce micro-batch size further:**
   ```python
   MAX_MICRO_BATCH_SIZE = 16  # or 8
   ```

2. **Enable gradient checkpointing** (if your model supports it):
   ```python
   model.gradient_checkpointing_enable()
   ```

3. **Use mixed precision:**
   ```python
   # Should already be enabled, but verify
   torch.cuda.amp.autocast()
   ```

### Slower Training

- Gradient accumulation adds overhead (~5-10%)
- More accumulation steps = more overhead
- Trade-off: memory vs. speed

### Different Results with Accumulation

- Should be mathematically identical
- Small differences (<1%) due to:
  - Floating point precision
  - Random number generator state
  - Batch normalization (if used)

---

## Example: Complete Workflow

```bash
# 1. Test gradient accumulation logic
python auto_gradient_accumulation.py

# 2. Run tests to ensure everything works
pytest test_*.py -v

# 3. Find your max micro-batch size
python test_max_batch.py  # Create this based on template above

# 4. Generate configs with automatic gradient accumulation
# Edit MAX_MICRO_BATCH_SIZE in generate_branch_configs_with_grad_accum.py
python generate_branch_configs_with_grad_accum.py

# 5. Review generated configs
ls configs_branch_sweep/
cat configs_branch_sweep/config_branch_k8.py

# 6. Run the sweep
python run_branch_sweep.py

# 7. Monitor
watch -n 1 nvidia-smi
tail -f logs_branch_sweep/branch_k8_*.log
```

---

## References

- **Critical Batch Size Theory:** https://arxiv.org/abs/1812.06162
- **Gradient Accumulation:** Used in all modern large-scale training
- **Scaling Laws:** Batch size scales linearly, LR scales as sqrt(k) for Adam

---

## Questions?

Common scenarios:

**Q: Do I need gradient accumulation for small models?**
A: Probably not if k ≤ 8. But it doesn't hurt to enable it.

**Q: How do I choose max_micro_batch_size?**
A: Either test empirically or use automatic estimation. Start conservative.

**Q: Will results be exactly the same with accumulation?**
A: Yes, mathematically equivalent. Tiny numerical differences are expected.

**Q: Does this work with other optimizers?**
A: Yes! Gradient accumulation is optimizer-agnostic.

**Q: Can I mix different micro-batch sizes across k-values?**
A: Yes, each k can have different micro-batch/accumulation settings.



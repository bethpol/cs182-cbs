"""
Automatic Gradient Accumulation for Batch Size Scaling

This module provides utilities to automatically determine gradient accumulation
settings when scaled batch sizes exceed GPU memory capacity.

Key Concept:
    effective_batch_size = micro_batch_size_per_gpu * num_gpus * gradient_accumulation_steps
    
    When scaling batch size by factor k, if the scaled batch size doesn't fit
    in memory, we keep micro_batch_size small and increase gradient_accumulation_steps.
    
    For multi-GPU training (data parallelism), each GPU processes a portion of the batch,
    so the effective batch size is multiplied by the number of GPUs.

Strategies:
    1. FIXED_MICRO_BATCH: User specifies max micro-batch size per GPU
    2. MEMORY_BASED: Estimate from GPU memory and model size
    3. BINARY_SEARCH: Run test batches to find max size (experimental)
"""

import math
import subprocess
import json
from typing import Optional, Tuple
from dataclasses import dataclass


@dataclass
class GradientAccumulationConfig:
    """Configuration for gradient accumulation with multi-GPU support."""
    micro_batch_size_per_gpu: int
    gradient_accumulation_steps: int
    num_gpus: int
    effective_batch_size: int
    
    def __str__(self):
        if self.num_gpus > 1:
            return (f"Micro-batch per GPU: {self.micro_batch_size_per_gpu}, "
                    f"GPUs: {self.num_gpus}, "
                    f"Grad accum: {self.gradient_accumulation_steps}, "
                    f"Effective: {self.effective_batch_size}")
        else:
            return (f"Micro-batch: {self.micro_batch_size_per_gpu}, "
                    f"Grad accum: {self.gradient_accumulation_steps}, "
                    f"Effective: {self.effective_batch_size}")


def get_num_gpus() -> int:
    """
    Detect the number of available GPUs.
    
    Returns:
        Number of GPUs (minimum 1)
    """
    try:
        # Try to get GPU count from nvidia-smi
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=count', '--format=csv,noheader'],
            capture_output=True,
            text=True,
            timeout=5
        )
        
        if result.returncode == 0:
            # nvidia-smi returns one line per GPU, so count lines
            lines = result.stdout.strip().split('\n')
            num_gpus = len([line for line in lines if line.strip()])
            return max(1, num_gpus)
    
    except (subprocess.TimeoutExpired, FileNotFoundError):
        pass
    
    # Try PyTorch if available
    try:
        import torch
        if torch.cuda.is_available():
            return max(1, torch.cuda.device_count())
    except ImportError:
        pass
    
    # Default to 1 GPU
    return 1


def calculate_gradient_accumulation(
    target_batch_size: int,
    max_micro_batch_size_per_gpu: int,
    num_gpus: int = 1,
    ensure_divisible: bool = True
) -> GradientAccumulationConfig:
    """
    Calculate gradient accumulation steps for a target batch size with multi-GPU support.
    
    The effective batch size formula with multi-GPU training is:
        effective_batch = micro_batch_per_gpu × num_gpus × grad_accum_steps
    
    Args:
        target_batch_size: The desired effective batch size (k * BASE_BATCH_SIZE)
        max_micro_batch_size_per_gpu: Maximum batch size per GPU that fits in memory
        num_gpus: Number of GPUs for data parallelism (default: 1)
        ensure_divisible: If True, adjust to ensure target_batch_size is divisible
                         by (micro_batch_per_gpu × num_gpus) for reproducibility
    
    Returns:
        GradientAccumulationConfig with calculated values
    
    Examples:
        >>> # Single GPU: Target 64, max 16 fits
        >>> config = calculate_gradient_accumulation(64, 16, num_gpus=1)
        >>> print(config)
        Micro-batch: 16, Grad accum: 4, Effective: 64
        
        >>> # Multi-GPU: Target 128, max 16 per GPU, 2 GPUs
        >>> config = calculate_gradient_accumulation(128, 16, num_gpus=2)
        >>> print(config)
        Micro-batch per GPU: 16, GPUs: 2, Grad accum: 4, Effective: 128
        >>> # (16 per GPU × 2 GPUs × 4 accum = 128)
    """
    # Total capacity without gradient accumulation
    max_batch_per_step = max_micro_batch_size_per_gpu * num_gpus
    
    if target_batch_size <= max_batch_per_step:
        # No gradient accumulation needed
        micro_batch_per_gpu = target_batch_size // num_gpus
        
        # Handle case where target is smaller than num_gpus
        if micro_batch_per_gpu == 0:
            micro_batch_per_gpu = 1
            # Adjust num_gpus used to match target as closely as possible
            effective_batch = min(target_batch_size, num_gpus)
        else:
            effective_batch = micro_batch_per_gpu * num_gpus
        
        return GradientAccumulationConfig(
            micro_batch_size_per_gpu=micro_batch_per_gpu,
            gradient_accumulation_steps=1,
            num_gpus=num_gpus,
            effective_batch_size=effective_batch
        )
    
    # Calculate how many accumulation steps we need
    grad_accum_steps = math.ceil(target_batch_size / max_batch_per_step)
    
    if ensure_divisible:
        # Adjust to ensure target is evenly divisible
        micro_batch_per_gpu = target_batch_size // (grad_accum_steps * num_gpus)
        
        # May need one more accumulation step if there's a remainder
        if target_batch_size % (grad_accum_steps * num_gpus) != 0:
            grad_accum_steps += 1
            micro_batch_per_gpu = target_batch_size // (grad_accum_steps * num_gpus)
    else:
        micro_batch_per_gpu = max_micro_batch_size_per_gpu
    
    effective_batch = micro_batch_per_gpu * num_gpus * grad_accum_steps
    
    return GradientAccumulationConfig(
        micro_batch_size_per_gpu=micro_batch_per_gpu,
        gradient_accumulation_steps=grad_accum_steps,
        num_gpus=num_gpus,
        effective_batch_size=effective_batch
    )


def estimate_max_batch_size_from_memory(
    model_params: int,
    sequence_length: int,
    gpu_memory_gb: float,
    memory_fraction: float = 0.7,
    bytes_per_param: float = 16.0,  # For mixed precision (float16 + float32)
    activation_multiplier: float = 0.5  # More realistic multiplier
) -> int:
    """
    Estimate maximum batch size that fits in GPU memory.
    
    This is a rough heuristic based on:
    - Model parameters (weights, gradients, optimizer states)
    - Activation memory (scales with batch size and sequence length)
    
    Args:
        model_params: Number of model parameters
        sequence_length: Sequence length (context window)
        gpu_memory_gb: Available GPU memory in GB
        memory_fraction: Fraction of GPU memory to use (default 0.7 for safety)
        bytes_per_param: Bytes per parameter (16 for mixed precision)
        activation_multiplier: Multiplier for activation memory (0.5 is reasonable default)
    
    Returns:
        Estimated maximum batch size
    
    Note:
        This is a conservative estimate. Actual capacity may vary based on:
        - Model architecture (attention is O(n²))
        - Framework overhead
        - Other processes using GPU
    """
    # Convert GPU memory to bytes
    available_memory = gpu_memory_gb * 1e9 * memory_fraction
    
    # Model memory (weights + gradients + optimizer states)
    # AdamW stores: params (fp32) + gradients (fp32) + m (fp32) + v (fp32) = 4x
    # Mixed precision: params (fp16 + fp32) + grad (fp16 + fp32) + m + v ≈ 16 bytes/param
    model_memory = model_params * bytes_per_param
    
    # Memory available for activations
    activation_memory_budget = available_memory - model_memory
    
    if activation_memory_budget <= 0:
        return 1  # Model barely fits, only batch size 1
    
    # Estimate activation memory per sample
    # For transformers: ~4-6 bytes per token per param (very rough)
    # activation_multiplier adjusts this
    activation_per_sample = model_params * sequence_length * activation_multiplier
    
    max_batch = int(activation_memory_budget / activation_per_sample)
    
    # Sanity bounds
    max_batch = max(1, min(max_batch, 1024))
    
    return max_batch


def get_gpu_memory() -> Optional[float]:
    """
    Query GPU memory using nvidia-smi.
    
    Returns:
        GPU memory in GB, or None if unable to query
    """
    try:
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=memory.total', '--format=csv,noheader,nounits'],
            capture_output=True,
            text=True,
            timeout=5
        )
        
        if result.returncode == 0:
            memory_mb = float(result.stdout.strip().split('\n')[0])
            return memory_mb / 1024  # Convert MB to GB
        
    except (subprocess.TimeoutExpired, FileNotFoundError, ValueError):
        pass
    
    return None


def auto_configure_batch_size(
    target_batch_size: int,
    model_params: int,
    sequence_length: int,
    max_micro_batch_size_per_gpu: Optional[int] = None,
    num_gpus: Optional[int] = None,
    gpu_memory_gb: Optional[float] = None
) -> GradientAccumulationConfig:
    """
    Automatically configure batch size and gradient accumulation with multi-GPU support.
    
    This is the main entry point for automatic configuration.
    
    Args:
        target_batch_size: Desired effective batch size
        model_params: Number of model parameters
        sequence_length: Sequence length
        max_micro_batch_size_per_gpu: If provided, use this as the max micro-batch size per GPU.
                                      If None, estimate from GPU memory.
        num_gpus: Number of GPUs. If None, will attempt to detect automatically.
        gpu_memory_gb: GPU memory in GB. If None, will attempt to query.
    
    Returns:
        GradientAccumulationConfig
    
    Example:
        >>> # Automatic configuration
        >>> config = auto_configure_batch_size(
        ...     target_batch_size=256,
        ...     model_params=45_000_000,
        ...     sequence_length=128,
        ...     num_gpus=2
        ... )
    """
    # Detect number of GPUs if not provided
    if num_gpus is None:
        num_gpus = get_num_gpus()
    
    if max_micro_batch_size_per_gpu is None:
        # Try to estimate from GPU memory
        if gpu_memory_gb is None:
            gpu_memory_gb = get_gpu_memory()
        
        if gpu_memory_gb is not None:
            max_micro_batch_size_per_gpu = estimate_max_batch_size_from_memory(
                model_params=model_params,
                sequence_length=sequence_length,
                gpu_memory_gb=gpu_memory_gb
            )
        else:
            # Fallback to conservative default
            max_micro_batch_size_per_gpu = 8
            print(f"Warning: Could not determine GPU memory. "
                  f"Using conservative default: {max_micro_batch_size_per_gpu}")
    
    return calculate_gradient_accumulation(
        target_batch_size=target_batch_size,
        max_micro_batch_size_per_gpu=max_micro_batch_size_per_gpu,
        num_gpus=num_gpus,
        ensure_divisible=True
    )


def print_batch_size_table(
    k_values: list,
    base_batch_size: int,
    max_micro_batch_size_per_gpu: int,
    num_gpus: int = 1
):
    """
    Print a table showing batch size configurations for different k values.
    
    Args:
        k_values: List of k scaling factors
        base_batch_size: Base batch size
        max_micro_batch_size_per_gpu: Maximum micro-batch size per GPU
        num_gpus: Number of GPUs (default: 1)
    """
    print("\nBatch Size Configuration Table")
    print("=" * 100)
    
    if num_gpus > 1:
        print(f"{'k':<4} | {'Target':<8} | {'Micro/GPU':<10} | "
              f"{'GPUs':<5} | {'Grad Accum':<11} | {'Effective':<10} | {'Formula':<30}")
        print("-" * 100)
    else:
        print(f"{'k':<4} | {'Target':<8} | {'Micro-Batch':<12} | "
              f"{'Grad Accum':<11} | {'Effective':<10}")
        print("-" * 100)
    
    for k in k_values:
        target = k * base_batch_size
        config = calculate_gradient_accumulation(target, max_micro_batch_size_per_gpu, num_gpus)
        
        if num_gpus > 1:
            formula = f"{config.micro_batch_size_per_gpu}×{num_gpus}×{config.gradient_accumulation_steps}"
            print(f"{k:<4} | {target:<8} | {config.micro_batch_size_per_gpu:<10} | "
                  f"{num_gpus:<5} | {config.gradient_accumulation_steps:<11} | "
                  f"{config.effective_batch_size:<10} | {formula:<30}")
        else:
            print(f"{k:<4} | {target:<8} | {config.micro_batch_size_per_gpu:<12} | "
                  f"{config.gradient_accumulation_steps:<11} | {config.effective_batch_size:<10}")
    
    print("=" * 100)


# =============================================================================
# Testing and examples
# =============================================================================

def test_basic_calculations():
    """Test basic gradient accumulation calculations."""
    print("\n" + "="*80)
    print("Test: Basic Gradient Accumulation Calculations")
    print("="*80)
    
    test_cases = [
        (64, 64),   # Fits exactly
        (64, 32),   # Need 2x accumulation
        (64, 16),   # Need 4x accumulation
        (100, 16),  # Non-divisible
    ]
    
    for target, max_micro in test_cases:
        config = calculate_gradient_accumulation(target, max_micro)
        print(f"\nTarget: {target}, Max Micro: {max_micro}")
        print(f"  → {config}")


def test_memory_estimation():
    """Test GPU memory estimation."""
    print("\n" + "="*80)
    print("Test: Memory-Based Batch Size Estimation")
    print("="*80)
    
    test_configs = [
        (45_000_000, 128, 16),  # 45M params, seq len 128, 16GB GPU
        (45_000_000, 128, 24),  # 45M params, seq len 128, 24GB GPU
        (124_000_000, 1024, 40),  # 124M params, seq len 1024, 40GB GPU
    ]
    
    for params, seq_len, gpu_gb in test_configs:
        max_batch = estimate_max_batch_size_from_memory(params, seq_len, gpu_gb)
        print(f"\nModel: {params/1e6:.0f}M params, Seq: {seq_len}, GPU: {gpu_gb}GB")
        print(f"  → Estimated max batch size: {max_batch}")


def test_full_pipeline():
    """Test the full auto-configuration pipeline."""
    print("\n" + "="*80)
    print("Test: Full Auto-Configuration Pipeline (Single GPU)")
    print("="*80)
    
    k_values = [1, 2, 4, 8, 16, 32]
    base_batch_size = 8
    model_params = 45_000_000
    sequence_length = 128
    gpu_memory_gb = 24  # A100 24GB or similar
    
    # Estimate max micro-batch size
    max_micro_batch = estimate_max_batch_size_from_memory(
        model_params, sequence_length, gpu_memory_gb
    )
    
    print(f"\nConfiguration:")
    print(f"  Base batch size: {base_batch_size}")
    print(f"  Model params: {model_params/1e6:.1f}M")
    print(f"  Sequence length: {sequence_length}")
    print(f"  GPU memory: {gpu_memory_gb}GB")
    print(f"  Number of GPUs: 1")
    print(f"  Estimated max micro-batch: {max_micro_batch}")
    
    print_batch_size_table(k_values, base_batch_size, max_micro_batch, num_gpus=1)


def test_multi_gpu():
    """Test multi-GPU configuration."""
    print("\n" + "="*80)
    print("Test: Multi-GPU Configuration")
    print("="*80)
    
    k_values = [1, 2, 4, 8, 16, 32]
    base_batch_size = 8
    max_micro_batch_per_gpu = 32  # What fits on each GPU
    num_gpus = 4
    
    print(f"\nConfiguration:")
    print(f"  Base batch size: {base_batch_size}")
    print(f"  Max micro-batch per GPU: {max_micro_batch_per_gpu}")
    print(f"  Number of GPUs: {num_gpus}")
    print(f"  Max batch per step (no accum): {max_micro_batch_per_gpu * num_gpus}")
    
    print_batch_size_table(k_values, base_batch_size, max_micro_batch_per_gpu, num_gpus)
    
    # Show specific examples
    print("\nExample calculations:")
    for k in [1, 8, 16]:
        target = k * base_batch_size
        config = calculate_gradient_accumulation(target, max_micro_batch_per_gpu, num_gpus)
        print(f"\nk={k}: Target batch = {target}")
        print(f"  → {config}")
        print(f"  → Formula: {config.micro_batch_size_per_gpu} (per GPU) × "
              f"{num_gpus} (GPUs) × {config.gradient_accumulation_steps} (accum) = "
              f"{config.effective_batch_size}")


if __name__ == "__main__":
    # Detect actual GPUs
    num_gpus_detected = get_num_gpus()
    print(f"Detected {num_gpus_detected} GPU(s)")
    
    test_basic_calculations()
    test_memory_estimation()
    test_full_pipeline()
    test_multi_gpu()


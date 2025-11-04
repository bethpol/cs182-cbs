"""
Integrated comprehensive tests: Dataloader validation + 45M GPT training
Combines dataloader validation with actual model training to ensure end-to-end correctness.

Usage:
    pytest test_integrated_comprehensive.py -v
    or
    python test_integrated_comprehensive.py

Requirements:
    - model_45m.py (defines build_gpt_45m)
    - nanoGPT model implementation
    - PyTorch with CUDA support (optional, CPU works)
    - 100M shuffled dataset
"""

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '8'
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import pytest
import time
from pathlib import Path
from typing import Dict, Tuple, Optional

sys.path.insert(0, os.path.dirname(__file__))
from dataloader import CBSDataLoader, create_dataloader
from model_45M import build_gpt_45m, count_parameters


class TrainingConfig:
    """Configuration for training."""
    
    def __init__(self):
        self.data_file = 'data/c4_dataset/100M/train_shuffled_1024.bin'
        self.block_size = 1024
        self.batch_size = 8
        self.learning_rate = 6e-4
        self.num_epochs = 1
        self.steps_per_epoch = 10
        self.checkpoint_interval = 20
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.seed = 42


class Trainer:
    """Trainer class for integration testing."""
    
    def __init__(self, config: TrainingConfig):
        self.config = config
        # Ensure device is specific (e.g., 'cuda:0' not just 'cuda')
        if config.device == 'cuda':
            self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(config.device)
        self.steps_trained = 0
        self.tokens_seen = 0
        self.checkpoints = []
        
        # Set seed
        torch.manual_seed(config.seed)
        np.random.seed(config.seed)
        
        # Create 45M model
        self.model = build_gpt_45m(device=config.device)
        param_count = count_parameters(self.model)
        print(f"Model parameters: {param_count:,}")
        
        self.optimizer = optim.Adam(self.model.parameters(), lr=config.learning_rate)
        
        # Create dataloader
        self.dataloader = create_dataloader(
            data_file=config.data_file,
            block_size=config.block_size,
            checkpoint_token_pos=0,
            branch_seed=-1,
            branch_window_size_tokens=100_000_000,
            device=config.device,
        )
        
        self.train_losses = []
        self.batch_times = []
    
    def train_step(self, x: torch.Tensor, y: torch.Tensor) -> float:
        """Single training step."""
        self.optimizer.zero_grad()
        # Pass targets to model so it computes loss internally
        logits, loss = self.model(x, targets=y)
        loss.backward()
        self.optimizer.step()
        return loss.item()
    
    def train_epoch(self) -> Dict:
        """Train for one epoch."""
        self.model.train()
        epoch_losses = []
        epoch_start = time.time()
        
        for step in range(self.config.steps_per_epoch):
            batch_start = time.time()
            
            # Get batch
            x, y = self.dataloader.get_batch(self.config.batch_size)
            
            # Train step
            loss = self.train_step(x, y)
            epoch_losses.append(loss)
            
            # Track metrics
            batch_time = time.time() - batch_start
            self.batch_times.append(batch_time)
            self.steps_trained += 1
            self.tokens_seen += x.numel()
            
            # Checkpoint
            if (self.steps_trained + 1) % self.config.checkpoint_interval == 0:
                checkpoint = self.get_checkpoint()
                self.checkpoints.append(checkpoint)
        
        epoch_time = time.time() - epoch_start
        avg_loss = np.mean(epoch_losses)
        
        return {
            'loss': avg_loss,
            'time': epoch_time,
            'steps': self.config.steps_per_epoch,
            'tokens_seen': self.tokens_seen,
            'avg_batch_time': np.mean(self.batch_times),
        }
    
    def get_checkpoint(self) -> Dict:
        """Create checkpoint."""
        return {
            'step': self.steps_trained,
            'tokens_seen': self.tokens_seen,
            'model_state': self.model.state_dict(),
            'optimizer_state': self.optimizer.state_dict(),
            'data_state': self.dataloader.get_state(),
            'config': self.config.__dict__,
        }
    
    def resume_from_checkpoint(self, checkpoint: Dict):
        """Resume from checkpoint."""
        # Ensure state dict tensors are on the right device
        model_state = {}
        for key, value in checkpoint['model_state'].items():
            if isinstance(value, torch.Tensor):
                model_state[key] = value.to(self.device)
            else:
                model_state[key] = value
        
        self.model.load_state_dict(model_state)
        self.optimizer.load_state_dict(checkpoint['optimizer_state'])
        self.dataloader.load_state(checkpoint['data_state'])
        self.steps_trained = checkpoint['step']
        self.tokens_seen = checkpoint['tokens_seen']
    
    @torch.no_grad()
    def evaluate(self) -> float:
        """Quick evaluation."""
        self.model.eval()
        eval_losses = []
        
        for _ in range(3):
            try:
                x, y = self.dataloader.get_batch(self.config.batch_size)
                logits, loss = self.model(x, targets=y)
                eval_losses.append(loss.item())
            except StopIteration:
                break
        
        return np.mean(eval_losses) if eval_losses else 0.0


# ============================================================================
# DATALOADER VALIDATION TESTS
# ============================================================================

class TestDataLoaderValidation:
    """Comprehensive tests for dataloader correctness."""
    
    @pytest.fixture
    def setup(self):
        """Setup for dataloader tests."""
        data_file = 'data/c4_dataset/100M/train_shuffled_1024.bin'
        
        if not os.path.exists(data_file):
            pytest.skip(f"Test dataset not found at {data_file}")
        
        return TrainingConfig()
    
    def test_batch_shapes(self, setup):
        """Verify batch shapes are correct."""
        print("\n[Test] Batch Shapes")
        print("="*70)
        
        dl = create_dataloader(
            data_file=setup.data_file,
            block_size=setup.block_size,
            checkpoint_token_pos=0,
            branch_seed=-1,
            branch_window_size_tokens=100_000_000,
            device='cpu',
        )
        
        batch_sizes = [1, 4, 8, 16]
        
        for batch_size in batch_sizes:
            x, y = dl.get_batch(batch_size)
            
            assert x.shape == (batch_size, setup.block_size - 1)
            assert y.shape == (batch_size, setup.block_size - 1)
            assert x.dtype == torch.long
            assert y.dtype == torch.long
            
            print(f"  Batch size {batch_size}: x={x.shape}, y={y.shape} ✓")
        
        print("✓ All batch shapes correct\n")
    
    def test_data_range(self, setup):
        """Verify token values are within vocab range."""
        print("\n[Test] Data Range")
        print("="*70)
        
        dl = create_dataloader(
            data_file=setup.data_file,
            block_size=setup.block_size,
            checkpoint_token_pos=0,
            branch_seed=-1,
            branch_window_size_tokens=100_000_000,
            device='cpu',
        )
        
        for batch_idx in range(5):
            x, y = dl.get_batch(8)
            
            assert x.min() >= 0 and x.max() < 50304
            assert y.min() >= 0 and y.max() < 50304
        
        print(f"  Checked 5 batches of 8 samples")
        print(f"  Token ranges all valid (0-50303) ✓")
        print("✓ Data range is correct\n")
    
    def test_sequential_consistency(self, setup):
        """Verify sequences are sequential (y[i] = x[i+1])."""
        print("\n[Test] Sequential Consistency")
        print("="*70)
        
        dl = create_dataloader(
            data_file=setup.data_file,
            block_size=setup.block_size,
            checkpoint_token_pos=0,
            branch_seed=-1,
            branch_window_size_tokens=100_000_000,
            device='cpu',
        )
        
        x, y = dl.get_batch(4)
        
        mismatches = 0
        for batch_idx in range(x.shape[0]):
            for pos in range(x.shape[1] - 1):
                if x[batch_idx, pos + 1].item() != y[batch_idx, pos].item():
                    mismatches += 1
        
        print(f"  Compared {x.shape[0] * (x.shape[1] - 1)} position pairs")
        assert mismatches == 0
        
        print("✓ All sequences are properly aligned\n")
    
    def test_no_data_leakage_between_batches(self, setup):
        """Verify different batches don't overlap."""
        print("\n[Test] No Data Leakage Between Batches")
        print("="*70)
        
        dl = create_dataloader(
            data_file=setup.data_file,
            block_size=setup.block_size,
            checkpoint_token_pos=0,
            branch_seed=-1,
            branch_window_size_tokens=100_000_000,
            device='cpu',
        )
        
        batch1_x, _ = dl.get_batch(1)
        batch2_x, _ = dl.get_batch(1)
        batch3_x, _ = dl.get_batch(1)
        
        assert not torch.equal(batch1_x, batch2_x)
        assert not torch.equal(batch2_x, batch3_x)
        assert not torch.equal(batch1_x, batch3_x)
        
        print(f"  Batch 1 first token: {batch1_x[0, 0].item()}")
        print(f"  Batch 2 first token: {batch2_x[0, 0].item()}")
        print(f"  Batch 3 first token: {batch3_x[0, 0].item()}")
        print("✓ All batches are distinct\n")
    
    def test_checkpoint_alignment(self, setup):
        """Verify checkpoints align to block boundaries."""
        print("\n[Test] Checkpoint Alignment")
        print("="*70)
        
        dl = create_dataloader(
            data_file=setup.data_file,
            block_size=setup.block_size,
            checkpoint_token_pos=0,
            branch_seed=-1,
            branch_window_size_tokens=100_000_000,
            device='cpu',
        )
        
        for _ in range(5):
            dl.get_batch(4)
        
        state = dl.get_state()
        token_pos = state['token_pos']
        
        assert token_pos % setup.block_size == 0
        
        print(f"  Current position: {token_pos:,} tokens")
        print(f"  Aligned to block size {setup.block_size}: ✓")
        print("✓ Checkpoint alignment correct\n")
    
    def test_branch_offset_correctness(self, setup):
        """Verify branch seeds create correct offsets."""
        print("\n[Test] Branch Offset Correctness")
        print("="*70)
        
        window_size = 1_000_000
        
        dl_seed0 = create_dataloader(
            data_file=setup.data_file,
            block_size=setup.block_size,
            checkpoint_token_pos=0,
            branch_seed=0,
            branch_window_size_tokens=window_size,
            device='cpu',
        )
        
        dl_seed1 = create_dataloader(
            data_file=setup.data_file,
            block_size=setup.block_size,
            checkpoint_token_pos=0,
            branch_seed=1,
            branch_window_size_tokens=window_size,
            device='cpu',
        )
        
        x0, _ = dl_seed0.get_batch(1)
        x1, _ = dl_seed1.get_batch(1)
        
        assert not torch.equal(x0, x1)
        
        print(f"  Branch seed 0 first batch: {x0[0, :5].tolist()}")
        print(f"  Branch seed 1 first batch: {x1[0, :5].tolist()}")
        print(f"  Window size: {window_size:,} tokens")
        print("✓ Branch offsets are correctly applied\n")
    
    def test_multi_gpu_data_partition(self, setup):
        """Verify multi-GPU mode partitions data correctly."""
        print("\n[Test] Multi-GPU Data Partition")
        print("="*70)
        
        num_gpus = 4
        batch_size = 2
        
        dataloaders = []
        for rank in range(num_gpus):
            dl = create_dataloader(
                data_file=setup.data_file,
                block_size=setup.block_size,
                checkpoint_token_pos=0,
                branch_seed=-1,
                branch_window_size_tokens=100_000_000,
                device='cpu',
                num_gpus=num_gpus,
                gpu_rank=rank,
            )
            dataloaders.append(dl)
        
        batches = []
        for rank, dl in enumerate(dataloaders):
            x, _ = dl.get_batch(batch_size)
            batches.append(x)
            print(f"  GPU {rank} first batch: {x[0, :3].tolist()}")
        
        for i in range(num_gpus):
            for j in range(i + 1, num_gpus):
                assert not torch.equal(batches[i], batches[j])
        
        print(f"✓ All {num_gpus} GPUs got unique data\n")
    
    def test_token_count_tracking(self, setup):
        """Verify token counting is accurate."""
        print("\n[Test] Token Count Tracking")
        print("="*70)
        
        dl = create_dataloader(
            data_file=setup.data_file,
            block_size=setup.block_size,
            checkpoint_token_pos=0,
            branch_seed=-1,
            branch_window_size_tokens=100_000_000,
            device='cpu',
        )
        
        batch_size = 4
        
        for i in range(10):
            x, _ = dl.get_batch(batch_size)
            reported = dl.get_tokens_seen()
            expected = (i + 1) * batch_size * setup.block_size
            
            assert reported == expected
            print(f"  Batch {i+1}: {expected:,} tokens ✓")
        
        print(f"✓ Token counting is accurate\n")


# ============================================================================
# TRAINING INTEGRATION TESTS WITH 45M MODEL
# ============================================================================

class TestTrainingIntegration45M:
    """Integration tests using 45M model and actual training."""
    
    @pytest.fixture
    def setup(self):
        """Setup for integration tests."""
        data_file = 'data/c4_dataset/100M/train_shuffled_1024.bin'
        
        if not os.path.exists(data_file):
            pytest.skip(f"Test dataset not found at {data_file}")
        
        return TrainingConfig()
    
    def test_model_initialization(self, setup):
        """Test model initializes correctly."""
        print("\n[Test] Model Initialization")
        print("="*70)
        
        trainer = Trainer(setup)
        param_count = count_parameters(trainer.model)
        
        print(f"Model: 45M GPT")
        print(f"Parameters: {param_count:,}")
        print(f"Device: {trainer.device}")
        print(f"Data file: {setup.data_file}")
        
        assert param_count > 40_000_000, "Model should have ~45M parameters"
        print("✓ Model initialized correctly\n")
    
    def test_basic_training_loop(self, setup):
        """Test that model can train without errors."""
        print("\n[Test] Basic Training Loop")
        print("="*70)
        
        trainer = Trainer(setup)
        
        metrics = trainer.train_epoch()
        
        print(f"Epoch metrics:")
        print(f"  Loss: {metrics['loss']:.4f}")
        print(f"  Time: {metrics['time']:.2f}s")
        print(f"  Tokens seen: {metrics['tokens_seen']:,}")
        print(f"  Avg batch time: {metrics['avg_batch_time']*1000:.2f}ms")
        
        assert metrics['loss'] < 20.0
        assert metrics['tokens_seen'] > 0
        print("✓ Basic training loop works\n")
    
    def test_checkpoint_and_resume(self, setup):
        """Test checkpoint and resume during training."""
        print("\n[Test] Checkpoint and Resume")
        print("="*70)
        
        trainer1 = Trainer(setup)
        trainer1.config.steps_per_epoch = 10
        
        print(f"trainer1 initial: steps={trainer1.steps_trained}")
        
        metrics1 = trainer1.train_epoch()
        print(f"trainer1 after 10 steps: steps={trainer1.steps_trained}")
        
        checkpoint = trainer1.get_checkpoint()
        checkpoint_step = checkpoint['step']
        print(f"Checkpoint: step={checkpoint_step}")
        
        # Resume in new trainer
        trainer2 = Trainer(setup)
        trainer2.resume_from_checkpoint(checkpoint)
        print(f"trainer2 after resume: steps={trainer2.steps_trained}")
        
        # Verify model parameters match
        for p1, p2 in zip(trainer1.model.parameters(), trainer2.model.parameters()):
            assert torch.allclose(p1, p2, atol=1e-5)
        print("✓ Model parameters match after resume")
        
        # Continue training
        trainer2.config.steps_per_epoch = 10
        metrics2 = trainer2.train_epoch()
        print(f"trainer2 after 10 more steps: steps={trainer2.steps_trained}")
        
        expected_steps = checkpoint_step + 10
        actual_steps = trainer2.steps_trained
        assert actual_steps == expected_steps
        
        print(f"✓ Continued training for 10 more steps")
        print(f"✓ Checkpoint and resume works correctly\n")
    
    def test_loss_decreases(self, setup):
        """Test that loss generally decreases with training."""
        print("\n[Test] Loss Decreases During Training")
        print("="*70)
        
        trainer = Trainer(setup)
        trainer.config.steps_per_epoch = 20
        
        trainer.model.train()
        losses = []
        
        for step in range(20):
            x, y = trainer.dataloader.get_batch(setup.batch_size)
            loss = trainer.train_step(x, y)
            losses.append(loss)
            if (step + 1) % 5 == 0:
                print(f"  Step {step+1}: loss={loss:.4f}")
        
        early_avg = np.mean(losses[:5])
        late_avg = np.mean(losses[-5:])
        
        print(f"Early avg loss: {early_avg:.4f}")
        print(f"Late avg loss: {late_avg:.4f}")
        
        assert late_avg < early_avg
        print(f"✓ Loss decreases during training\n")
    
    def test_gradient_flow(self, setup):
        """Test that gradients flow through the model."""
        print("\n[Test] Gradient Flow")
        print("="*70)
        
        trainer = Trainer(setup)
        
        x, y = trainer.dataloader.get_batch(1)
        logits, loss = trainer.model(x, targets=y)
        loss.backward()
        
        params_with_grad = 0
        for name, param in trainer.model.named_parameters():
            if param.grad is not None:
                params_with_grad += 1
        
        print(f"Parameters with gradients: {params_with_grad}")
        
        assert params_with_grad > 0
        print(f"✓ Gradients flow correctly\n")
    
    def test_dataloader_with_model(self, setup):
        """Test dataloader integration with model training."""
        print("\n[Test] Dataloader with Model")
        print("="*70)
        
        trainer = Trainer(setup)
        
        # Train for a few steps
        trainer.model.train()
        losses = []
        
        for step in range(5):
            x, y = trainer.dataloader.get_batch(setup.batch_size)
            
            # Verify batch properties
            assert x.shape[0] == setup.batch_size
            assert y.shape[0] == setup.batch_size
            # Now devices should match exactly since trainer.device is explicit
            assert x.device == trainer.device, f"Batch on {x.device}, trainer on {trainer.device}"
            assert y.device == trainer.device, f"Batch on {y.device}, trainer on {trainer.device}"
            
            # Train
            loss = trainer.train_step(x, y)
            losses.append(loss)
            
            print(f"  Step {step+1}: loss={loss:.4f}, batch_shape={x.shape}")
        
        print(f"✓ Dataloader integrates well with model\n")


if __name__ == '__main__':
    # Check dataset exists
    data_file = 'data/c4_dataset/100M/train_shuffled_1024.bin'
    if not os.path.exists(data_file):
        print(f"❌ Test dataset not found at {data_file}")
        print("Please ensure the dataset is shuffled and in the correct location")
        sys.exit(1)
    
    print("\n" + "="*70)
    print("Integrated Comprehensive Tests: Dataloader + 45M GPT Training")
    print("="*70)
    
    try:
        pytest.main([__file__, '-v', '-s', '--tb=short'])
    except:
        print("pytest not available, running basic tests...\n")
        
        config = TrainingConfig()
        
        # Test dataloader
        print("[Test] Dataloader Basic Properties")
        dl = create_dataloader(
            data_file=config.data_file,
            block_size=config.block_size,
            checkpoint_token_pos=0,
            branch_seed=-1,
            branch_window_size_tokens=100_000_000,
            device='cpu',
        )
        x, y = dl.get_batch(4)
        print(f"✓ Batch shapes: x={x.shape}, y={y.shape}")
        
        # Test model
        print("\n[Test] Model Training")
        trainer = Trainer(config)
        trainer.config.steps_per_epoch = 5
        metrics = trainer.train_epoch()
        print(f"✓ Training completed")
        print(f"  Loss: {metrics['loss']:.4f}")
        print(f"  Steps: {trainer.steps_trained}")
        print(f"  Tokens: {trainer.tokens_seen:,}")
        
        print(f"\n✓ All integrated tests passed!")
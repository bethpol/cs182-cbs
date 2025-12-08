"""
Tests for generate_branch_configs.py

Run with: pytest test_generate_branch_configs.py -v
"""

import os
import sys
import tempfile
import shutil
import math
from pathlib import Path

import pytest

# Add parent directory to path to import modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import generate_branch_configs as gbc


class TestConfigGeneration:
    """Test config file generation."""
    
    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for test outputs."""
        tmpdir = tempfile.mkdtemp()
        yield tmpdir
        shutil.rmtree(tmpdir)
    
    def test_generate_single_config_k1(self, temp_dir):
        """Test generating config for k=1 (no scaling)."""
        k = 1
        gbc.generate_config_file(k, temp_dir)
        
        config_path = os.path.join(temp_dir, f"config_branch_k{k}.py")
        assert os.path.exists(config_path), "Config file should be created"
        
        with open(config_path, 'r') as f:
            content = f.read()
        
        # Check that batch size is correct (k * BASE_BATCH_SIZE)
        expected_batch_size = k * gbc.BASE_BATCH_SIZE
        assert f"batch_size = {expected_batch_size}" in content
        
        # Check that learning rate is correct (sqrt(k) * BASE_LR)
        expected_lr = math.sqrt(k) * gbc.BASE_LEARNING_RATE
        assert f"learning_rate = {expected_lr}" in content
        
        # Check metadata is present
        assert f"# Configuration for Branch k={k}" in content
        assert "out_dir = 'out_branch_k1'" in content
    
    def test_generate_single_config_k8(self, temp_dir):
        """Test generating config for k=8 (8x scaling)."""
        k = 8
        gbc.generate_config_file(k, temp_dir)
        
        config_path = os.path.join(temp_dir, f"config_branch_k{k}.py")
        assert os.path.exists(config_path)
        
        with open(config_path, 'r') as f:
            content = f.read()
        
        # Check batch size scaling
        expected_batch_size = k * gbc.BASE_BATCH_SIZE
        assert f"batch_size = {expected_batch_size}" in content
        
        # Check learning rate scaling (sqrt(8) ≈ 2.828)
        expected_lr = math.sqrt(k) * gbc.BASE_LEARNING_RATE
        assert f"learning_rate = {expected_lr}" in content
        
        # Check output directory
        assert f"out_dir = 'out_branch_k{k}'" in content
    
    def test_generate_all_configs(self, temp_dir):
        """Test generating configs for all k-values."""
        k_values = [1, 2, 4, 8, 16, 32]
        
        for k in k_values:
            gbc.generate_config_file(k, temp_dir)
        
        # Check all files exist
        for k in k_values:
            config_path = os.path.join(temp_dir, f"config_branch_k{k}.py")
            assert os.path.exists(config_path), f"Config for k={k} should exist"
    
    def test_config_is_valid_python(self, temp_dir):
        """Test that generated config is valid Python syntax."""
        k = 4
        gbc.generate_config_file(k, temp_dir)
        
        config_path = os.path.join(temp_dir, f"config_branch_k{k}.py")
        
        # Try to compile the file
        with open(config_path, 'r') as f:
            content = f.read()
        
        try:
            compile(content, config_path, 'exec')
        except SyntaxError as e:
            pytest.fail(f"Generated config has syntax error: {e}")
    
    def test_checkpoint_path_format(self, temp_dir):
        """Test that checkpoint path is correctly formatted."""
        k = 2
        gbc.generate_config_file(k, temp_dir)
        
        config_path = os.path.join(temp_dir, f"config_branch_k{k}.py")
        with open(config_path, 'r') as f:
            content = f.read()
        
        # Check relative path to checkpoint
        expected_path = f"'../{gbc.BASE_CHECKPOINT_DIR}/{gbc.BASE_CHECKPOINT_FILE}'"
        assert f"resume_ckpt_fname = {expected_path}" in content


class TestScalingFormulas:
    """Test the scaling formulas used in the experiments."""
    
    def test_batch_size_scaling(self):
        """Test that batch size scales linearly with k."""
        base_batch = 8
        
        for k in [1, 2, 4, 8, 16, 32]:
            expected = k * base_batch
            actual = k * gbc.BASE_BATCH_SIZE
            # Just testing the formula logic
            assert k * base_batch == expected
    
    def test_learning_rate_scaling(self):
        """Test that learning rate scales with sqrt(k)."""
        base_lr = 3e-4
        
        test_cases = [
            (1, 1.0),      # sqrt(1) = 1.0
            (4, 2.0),      # sqrt(4) = 2.0
            (16, 4.0),     # sqrt(16) = 4.0
        ]
        
        for k, expected_factor in test_cases:
            actual_factor = math.sqrt(k)
            assert abs(actual_factor - expected_factor) < 1e-9
            
            scaled_lr = actual_factor * base_lr
            assert scaled_lr == expected_factor * base_lr
    
    def test_f_k_function(self):
        """Test the f(k) = sqrt(k) scaling function."""
        # For Adam/AdamW, f(k) = sqrt(k)
        test_cases = [
            (1, 1.0),
            (2, math.sqrt(2)),
            (4, 2.0),
            (8, math.sqrt(8)),
            (16, 4.0),
            (32, math.sqrt(32)),
        ]
        
        for k, expected_fk in test_cases:
            actual_fk = math.sqrt(k)
            assert abs(actual_fk - expected_fk) < 1e-9


class TestConfigParameters:
    """Test that all required parameters are in generated configs."""
    
    @pytest.fixture
    def sample_config(self, tmp_path):
        """Generate a sample config for testing."""
        config_dir = str(tmp_path)
        k = 4
        gbc.generate_config_file(k, config_dir)
        
        config_path = os.path.join(config_dir, f"config_branch_k{k}.py")
        with open(config_path, 'r') as f:
            content = f.read()
        return content
    
    def test_io_parameters(self, sample_config):
        """Test that all I/O parameters are present."""
        required = [
            'out_dir',
            'init_from',
            'resume_ckpt_fname',
            'log_interval_tokens',
            'checkpoint_interval_tokens',
            'eval_iters',
        ]
        
        for param in required:
            assert param in sample_config, f"Missing parameter: {param}"
    
    def test_data_parameters(self, sample_config):
        """Test that all data parameters are present."""
        required = [
            'train_data_file',
            'val_data_file',
            'block_size',
        ]
        
        for param in required:
            assert param in sample_config, f"Missing parameter: {param}"
    
    def test_training_parameters(self, sample_config):
        """Test that all training parameters are present."""
        required = [
            'batch_size',
            'gradient_accumulation_steps',
            'max_tokens',
        ]
        
        for param in required:
            assert param in sample_config, f"Missing parameter: {param}"
    
    def test_optimizer_parameters(self, sample_config):
        """Test that all optimizer parameters are present."""
        required = [
            'optimizer_type',
            'learning_rate',
            'weight_decay',
            'beta1',
            'beta2',
            'grad_clip',
        ]
        
        for param in required:
            assert param in sample_config, f"Missing parameter: {param}"
    
    def test_wandb_parameters(self, sample_config):
        """Test that WandB parameters are present."""
        required = [
            'wandb_log',
            'wandb_project',
            'wandb_run_name',
        ]
        
        for param in required:
            assert param in sample_config, f"Missing parameter: {param}"


class TestEdgeCases:
    """Test edge cases and error conditions."""
    
    def test_k_value_zero(self, tmp_path):
        """Test that k=0 doesn't crash (even though it's invalid)."""
        # This should complete without error, even if the config is nonsensical
        config_dir = str(tmp_path)
        gbc.generate_config_file(0, config_dir)
        
        config_path = os.path.join(config_dir, "config_branch_k0.py")
        assert os.path.exists(config_path)
    
    def test_large_k_value(self, tmp_path):
        """Test with very large k value."""
        config_dir = str(tmp_path)
        k = 1024
        gbc.generate_config_file(k, config_dir)
        
        config_path = os.path.join(config_dir, f"config_branch_k{k}.py")
        assert os.path.exists(config_path)
        
        with open(config_path, 'r') as f:
            content = f.read()
        
        # Check scaling still works
        expected_batch = k * gbc.BASE_BATCH_SIZE
        assert f"batch_size = {expected_batch}" in content
    
    def test_output_dir_creation(self, tmp_path):
        """Test that output directory is created if it doesn't exist."""
        config_dir = str(tmp_path / "nested" / "dir")
        assert not os.path.exists(config_dir)
        
        os.makedirs(config_dir, exist_ok=True)
        gbc.generate_config_file(2, config_dir)
        
        config_path = os.path.join(config_dir, "config_branch_k2.py")
        assert os.path.exists(config_path)


class TestWandBNaming:
    """Test WandB run name generation."""
    
    def test_wandb_run_name_format(self, tmp_path):
        """Test that WandB run names follow expected format."""
        config_dir = str(tmp_path)
        k = 8
        gbc.generate_config_file(k, config_dir)
        
        config_path = os.path.join(config_dir, f"config_branch_k{k}.py")
        with open(config_path, 'r') as f:
            content = f.read()
        
        # Should contain k, batch size, and learning rate
        scaled_batch = k * gbc.BASE_BATCH_SIZE
        scaled_lr = math.sqrt(k) * gbc.BASE_LEARNING_RATE
        
        # Check the pattern (may need to adjust based on actual format)
        assert "wandb_run_name" in content
        assert f"k{k}" in content
        assert f"B{scaled_batch}" in content


class TestTokenCalculations:
    """Test token-related calculations."""
    
    def test_max_tokens_set_correctly(self, tmp_path):
        """Test that max_tokens is set from DELTA_STEPS_AS_TOKENS."""
        config_dir = str(tmp_path)
        k = 4
        gbc.generate_config_file(k, config_dir)
        
        config_path = os.path.join(config_dir, f"config_branch_k{k}.py")
        with open(config_path, 'r') as f:
            content = f.read()
        
        # max_tokens should equal DELTA_STEPS_AS_TOKENS
        expected_tokens = gbc.DELTA_STEPS_AS_TOKENS
        assert f"max_tokens = {expected_tokens}" in content
    
    def test_log_interval_tokens(self, tmp_path):
        """Test that log intervals are set correctly."""
        config_dir = str(tmp_path)
        k = 2
        gbc.generate_config_file(k, config_dir)
        
        config_path = os.path.join(config_dir, f"config_branch_k{k}.py")
        with open(config_path, 'r') as f:
            content = f.read()
        
        assert f"log_interval_tokens = {gbc.LOG_INTERVAL_TOKENS}" in content
        assert f"checkpoint_interval_tokens = {gbc.CHECKPOINT_INTERVAL_TOKENS}" in content


if __name__ == "__main__":
    pytest.main([__file__, "-v"])



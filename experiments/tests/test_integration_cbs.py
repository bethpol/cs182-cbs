"""
Integration Tests for CBS Branching Experiments

These tests verify the entire system works end-to-end, including:
- Config generation with different GPU counts
- Correct batch size calculations for CBS experiments
- Multi-GPU consistency
- Generated config file validity
- Actual CBS experiment setup (BASE_BATCH_SIZE=32)

Run with: pytest test_integration_cbs.py -v
"""

import os
import sys
import tempfile
import shutil
from pathlib import Path

import pytest

# Add parent directory to path to import modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import auto_gradient_accumulation as aga


class TestCBSConfigGeneration:
    """Test config generation for CBS experiments."""
    
    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for test outputs."""
        tmpdir = tempfile.mkdtemp()
        yield tmpdir
        shutil.rmtree(tmpdir)
    
    def test_cbs_base_32_single_gpu(self):
        """Test CBS experiments with BASE_BATCH_SIZE=32 on single GPU."""
        base_batch = 32
        k_values = [1, 2, 4, 8, 16, 32]
        max_micro = 32
        num_gpus = 1
        
        # Generate configs for each k-value
        configs = {}
        for k in k_values:
            target = k * base_batch
            config = aga.calculate_gradient_accumulation(target, max_micro, num_gpus)
            configs[k] = config
        
        # Verify all effective batches match targets
        for k in k_values:
            target = k * base_batch
            assert configs[k].effective_batch_size == target, \
                f"k={k}: effective batch {configs[k].effective_batch_size} != target {target}"
        
        # Verify specific configurations
        assert configs[1].micro_batch_size_per_gpu == 32
        assert configs[1].gradient_accumulation_steps == 1
        
        assert configs[2].micro_batch_size_per_gpu == 32
        assert configs[2].gradient_accumulation_steps == 2
        
        assert configs[32].micro_batch_size_per_gpu == 32
        assert configs[32].gradient_accumulation_steps == 32
    
    def test_cbs_base_32_two_gpus(self):
        """Test CBS experiments with BASE_BATCH_SIZE=32 on two GPUs."""
        base_batch = 32
        k_values = [1, 2, 4, 8, 16, 32]
        max_micro = 32
        num_gpus = 2
        
        configs = {}
        for k in k_values:
            target = k * base_batch
            config = aga.calculate_gradient_accumulation(target, max_micro, num_gpus)
            configs[k] = config
        
        # Verify all effective batches match targets
        for k in k_values:
            target = k * base_batch
            assert configs[k].effective_batch_size == target
            assert configs[k].num_gpus == 2
        
        # Verify specific configurations
        # k=1: 16 per GPU × 2 GPUs = 32
        assert configs[1].micro_batch_size_per_gpu == 16
        assert configs[1].gradient_accumulation_steps == 1
        
        # k=2: 32 per GPU × 2 GPUs = 64
        assert configs[2].micro_batch_size_per_gpu == 32
        assert configs[2].gradient_accumulation_steps == 1
        
        # k=32: 32 per GPU × 2 GPUs × 16 accum = 1024
        assert configs[32].micro_batch_size_per_gpu == 32
        assert configs[32].gradient_accumulation_steps == 16
    
    def test_cbs_consistency_across_gpus(self):
        """Test that CBS configs are consistent across different GPU counts."""
        base_batch = 32
        k_values = [1, 2, 4, 8, 16, 32]
        max_micro = 32
        
        # Test 1, 2, 4 GPUs
        for num_gpus in [1, 2, 4]:
            for k in k_values:
                target = k * base_batch
                config = aga.calculate_gradient_accumulation(target, max_micro, num_gpus)
                
                assert config.effective_batch_size == target, \
                    f"GPU={num_gpus}, k={k}: effective {config.effective_batch_size} != target {target}"
    
    def test_cbs_batch_size_range(self):
        """Test the full CBS batch size range (32 to 1024)."""
        base_batch = 32
        k_values = [1, 2, 4, 8, 16, 32]
        expected_batches = [32, 64, 128, 256, 512, 1024]
        max_micro = 32
        
        for k, expected_batch in zip(k_values, expected_batches):
            target = k * base_batch
            assert target == expected_batch, f"k={k}: target {target} != expected {expected_batch}"
            
            # Test on single GPU
            config = aga.calculate_gradient_accumulation(target, max_micro, 1)
            assert config.effective_batch_size == expected_batch


class TestMultiGPUConsistency:
    """Test that multi-GPU configurations produce consistent results."""
    
    def test_effective_batch_matches_1_vs_2_gpus(self):
        """Test that 1 GPU and 2 GPU configs have same effective batch."""
        base_batch = 32
        k_values = [1, 2, 4, 8, 16, 32]
        max_micro = 32
        
        for k in k_values:
            target = k * base_batch
            
            config_1gpu = aga.calculate_gradient_accumulation(target, max_micro, 1)
            config_2gpu = aga.calculate_gradient_accumulation(target, max_micro, 2)
            
            # Effective batches must match
            assert config_1gpu.effective_batch_size == config_2gpu.effective_batch_size == target, \
                f"k={k}: 1GPU={config_1gpu.effective_batch_size}, " \
                f"2GPU={config_2gpu.effective_batch_size}, target={target}"
    
    def test_gradient_accumulation_reduces_with_more_gpus(self):
        """Test that more GPUs means less gradient accumulation needed."""
        target = 256  # k=8 with base=32
        max_micro = 32
        
        config_1gpu = aga.calculate_gradient_accumulation(target, max_micro, 1)
        config_2gpu = aga.calculate_gradient_accumulation(target, max_micro, 2)
        config_4gpu = aga.calculate_gradient_accumulation(target, max_micro, 4)
        
        # All should have same effective batch
        assert config_1gpu.effective_batch_size == 256
        assert config_2gpu.effective_batch_size == 256
        assert config_4gpu.effective_batch_size == 256
        
        # Gradient accumulation should decrease
        assert config_1gpu.gradient_accumulation_steps >= config_2gpu.gradient_accumulation_steps
        assert config_2gpu.gradient_accumulation_steps >= config_4gpu.gradient_accumulation_steps
    
    def test_formula_consistency(self):
        """Test that the formula micro × gpus × accum = effective holds."""
        base_batch = 32
        k_values = [1, 2, 4, 8, 16, 32]
        max_micro = 32
        
        for num_gpus in [1, 2, 4, 8]:
            for k in k_values:
                target = k * base_batch
                config = aga.calculate_gradient_accumulation(target, max_micro, num_gpus)
                
                calculated_effective = (config.micro_batch_size_per_gpu * 
                                       config.num_gpus * 
                                       config.gradient_accumulation_steps)
                
                assert calculated_effective == config.effective_batch_size, \
                    f"Formula broken: {config.micro_batch_size_per_gpu}×{config.num_gpus}×" \
                    f"{config.gradient_accumulation_steps} = {calculated_effective} != " \
                    f"{config.effective_batch_size}"


class TestConfigFileGeneration:
    """Test actual config file generation."""
    
    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory."""
        tmpdir = tempfile.mkdtemp()
        yield tmpdir
        shutil.rmtree(tmpdir)
    
    def test_generate_config_content(self, temp_dir):
        """Test that generated config files have correct content."""
        # Import the config generator
        import generate_branch_configs_with_grad_accum as gen
        
        # Test parameters
        k = 8
        base_batch = 32
        target = k * base_batch  # 256
        max_micro = 32
        num_gpus = 2
        
        # Calculate gradient accumulation
        config = aga.calculate_gradient_accumulation(target, max_micro, num_gpus)
        
        # Generate config
        gen.generate_config_file(k, temp_dir, config)
        
        # Check file exists
        config_file = os.path.join(temp_dir, f"config_branch_k{k}.py")
        assert os.path.exists(config_file), "Config file should be created"
        
        # Read and verify content
        with open(config_file, 'r') as f:
            content = f.read()
        
        # Check key values are present
        assert f"batch_size = {config.micro_batch_size_per_gpu}" in content
        assert f"gradient_accumulation_steps = {config.gradient_accumulation_steps}" in content
        assert f"k={k}" in content
        assert "Effective batch" in content or "effective batch" in content
        
        # Verify it's valid Python
        try:
            compile(content, config_file, 'exec')
        except SyntaxError as e:
            pytest.fail(f"Generated config has syntax error: {e}")
    
    def test_all_k_values_generate(self, temp_dir):
        """Test that configs generate for all k-values."""
        import generate_branch_configs_with_grad_accum as gen
        
        base_batch = 32
        k_values = [1, 2, 4, 8, 16, 32]
        max_micro = 32
        num_gpus = 1
        
        for k in k_values:
            target = k * base_batch
            config = aga.calculate_gradient_accumulation(target, max_micro, num_gpus)
            gen.generate_config_file(k, temp_dir, config)
        
        # Check all files exist
        for k in k_values:
            config_file = os.path.join(temp_dir, f"config_branch_k{k}.py")
            assert os.path.exists(config_file), f"Config for k={k} should exist"


class TestGPUDetection:
    """Test GPU detection functionality."""
    
    def test_get_num_gpus_returns_valid(self):
        """Test that GPU detection returns a valid number."""
        num_gpus = aga.get_num_gpus()
        
        assert isinstance(num_gpus, int)
        assert num_gpus >= 1, "Should detect at least 1 GPU or default to 1"
        assert num_gpus <= 16, "Sanity check: unlikely to have more than 16 GPUs"
    
    def test_gpu_memory_query(self):
        """Test GPU memory query."""
        memory = aga.get_gpu_memory()
        
        if memory is not None:
            assert isinstance(memory, float)
            assert memory > 0
            assert memory < 1000, "GPU memory should be reasonable (in GB)"


class TestEndToEndWorkflow:
    """Test the complete end-to-end workflow."""
    
    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory."""
        tmpdir = tempfile.mkdtemp()
        yield tmpdir
        shutil.rmtree(tmpdir)
    
    def test_complete_cbs_workflow_single_gpu(self, temp_dir):
        """Test complete CBS workflow for single GPU user."""
        import generate_branch_configs_with_grad_accum as gen
        
        # Step 1: Detect GPUs (simulate single GPU)
        num_gpus = 1
        
        # Step 2: Set parameters (CBS with base=32)
        base_batch = 32
        k_values = [1, 2, 4, 8, 16, 32]
        max_micro = 32
        
        # Step 3: Generate all configs
        all_effective_batches = []
        for k in k_values:
            target = k * base_batch
            config = aga.calculate_gradient_accumulation(target, max_micro, num_gpus)
            gen.generate_config_file(k, temp_dir, config)
            all_effective_batches.append((k, config.effective_batch_size))
        
        # Step 4: Verify all configs
        expected = [(1, 32), (2, 64), (4, 128), (8, 256), (16, 512), (32, 1024)]
        assert all_effective_batches == expected, \
            f"Effective batches don't match: got {all_effective_batches}, expected {expected}"
        
        # Step 5: Verify files are valid Python
        for k in k_values:
            config_file = os.path.join(temp_dir, f"config_branch_k{k}.py")
            assert os.path.exists(config_file)
            
            with open(config_file, 'r') as f:
                content = f.read()
            
            # Should compile without errors
            compile(content, config_file, 'exec')
    
    def test_complete_cbs_workflow_two_gpus(self, temp_dir):
        """Test complete CBS workflow for two GPU user."""
        import generate_branch_configs_with_grad_accum as gen
        
        num_gpus = 2
        base_batch = 32
        k_values = [1, 2, 4, 8, 16, 32]
        max_micro = 32
        
        all_effective_batches = []
        for k in k_values:
            target = k * base_batch
            config = aga.calculate_gradient_accumulation(target, max_micro, num_gpus)
            gen.generate_config_file(k, temp_dir, config)
            all_effective_batches.append((k, config.effective_batch_size))
        
        # Should have same effective batches as single GPU
        expected = [(1, 32), (2, 64), (4, 128), (8, 256), (16, 512), (32, 1024)]
        assert all_effective_batches == expected
        
        # All configs should exist and be valid
        for k in k_values:
            config_file = os.path.join(temp_dir, f"config_branch_k{k}.py")
            assert os.path.exists(config_file)
    
    def test_configs_differ_appropriately_between_gpu_counts(self, temp_dir):
        """Test that configs differ in implementation but match in results."""
        import generate_branch_configs_with_grad_accum as gen
        
        k = 8
        base_batch = 32
        target = k * base_batch  # 256
        max_micro = 32
        
        # Generate config for 1 GPU
        config_1gpu = aga.calculate_gradient_accumulation(target, max_micro, 1)
        temp_dir_1gpu = os.path.join(temp_dir, "1gpu")
        os.makedirs(temp_dir_1gpu)
        gen.generate_config_file(k, temp_dir_1gpu, config_1gpu)
        
        # Generate config for 2 GPUs
        config_2gpu = aga.calculate_gradient_accumulation(target, max_micro, 2)
        temp_dir_2gpu = os.path.join(temp_dir, "2gpu")
        os.makedirs(temp_dir_2gpu)
        gen.generate_config_file(k, temp_dir_2gpu, config_2gpu)
        
        # Read both configs
        with open(os.path.join(temp_dir_1gpu, f"config_branch_k{k}.py"), 'r') as f:
            content_1gpu = f.read()
        
        with open(os.path.join(temp_dir_2gpu, f"config_branch_k{k}.py"), 'r') as f:
            content_2gpu = f.read()
        
        # Configs should differ (different implementation)
        assert content_1gpu != content_2gpu, "Configs should differ in implementation"
        
        # But effective batch sizes should match
        assert config_1gpu.effective_batch_size == config_2gpu.effective_batch_size == 256


class TestEdgeCases:
    """Test edge cases and boundary conditions."""
    
    def test_very_large_batch_k32(self):
        """Test k=32 with batch size 1024."""
        base_batch = 32
        k = 32
        target = k * base_batch  # 1024
        max_micro = 32
        
        # Should work for both 1 and 2 GPUs
        config_1gpu = aga.calculate_gradient_accumulation(target, max_micro, 1)
        config_2gpu = aga.calculate_gradient_accumulation(target, max_micro, 2)
        
        assert config_1gpu.effective_batch_size == 1024
        assert config_2gpu.effective_batch_size == 1024
        
        # 1 GPU should need 32× accumulation
        assert config_1gpu.gradient_accumulation_steps == 32
        
        # 2 GPUs should need 16× accumulation
        assert config_2gpu.gradient_accumulation_steps == 16
    
    def test_minimum_batch_k1(self):
        """Test k=1 with batch size 32."""
        base_batch = 32
        k = 1
        target = k * base_batch  # 32
        max_micro = 32
        
        # Should fit perfectly on single GPU
        config = aga.calculate_gradient_accumulation(target, max_micro, 1)
        
        assert config.effective_batch_size == 32
        assert config.micro_batch_size_per_gpu == 32
        assert config.gradient_accumulation_steps == 1  # No accumulation needed
    
    def test_odd_gpu_counts(self):
        """Test with odd GPU counts (3, 5, 7)."""
        base_batch = 32
        k = 8
        target = k * base_batch  # 256
        max_micro = 32
        
        for num_gpus in [3, 5, 7]:
            config = aga.calculate_gradient_accumulation(target, max_micro, num_gpus)
            
            # Should still produce reasonable effective batch
            assert config.effective_batch_size > 0
            assert config.effective_batch_size <= target
            # Should be close to target (within one micro-batch per GPU)
            assert config.effective_batch_size >= target - (max_micro * num_gpus)


class TestRealWorldScenarios:
    """Test realistic scenarios that users will encounter."""
    
    def test_group_member_scenario(self):
        """Test typical group member scenario: 1 GPU, 24GB, CBS base=32."""
        num_gpus = 1
        base_batch = 32
        k_values = [1, 2, 4, 8, 16, 32]
        max_micro = 32  # What fits on 24GB GPU
        
        print("\nGroup Member Scenario (1 GPU):")
        for k in k_values:
            target = k * base_batch
            config = aga.calculate_gradient_accumulation(target, max_micro, num_gpus)
            
            print(f"  k={k:2d}: target={target:4d}, config={config.micro_batch_size_per_gpu}×{config.gradient_accumulation_steps}, "
                  f"effective={config.effective_batch_size}")
            
            assert config.effective_batch_size == target
    
    def test_two_gpu_user_scenario(self):
        """Test user with 2 GPUs scenario."""
        num_gpus = 2
        base_batch = 32
        k_values = [1, 2, 4, 8, 16, 32]
        max_micro = 32
        
        print("\nTwo GPU User Scenario:")
        for k in k_values:
            target = k * base_batch
            config = aga.calculate_gradient_accumulation(target, max_micro, num_gpus)
            
            print(f"  k={k:2d}: target={target:4d}, config={config.micro_batch_size_per_gpu}×{num_gpus}×{config.gradient_accumulation_steps}, "
                  f"effective={config.effective_batch_size}")
            
            assert config.effective_batch_size == target
    
    def test_results_match_across_scenarios(self):
        """Test that both scenarios produce matching effective batches."""
        base_batch = 32
        k_values = [1, 2, 4, 8, 16, 32]
        max_micro = 32
        
        print("\nCross-Scenario Consistency:")
        for k in k_values:
            target = k * base_batch
            
            config_1gpu = aga.calculate_gradient_accumulation(target, max_micro, 1)
            config_2gpu = aga.calculate_gradient_accumulation(target, max_micro, 2)
            
            match = "✓" if config_1gpu.effective_batch_size == config_2gpu.effective_batch_size else "✗"
            print(f"  k={k:2d}: 1GPU={config_1gpu.effective_batch_size:4d}, "
                  f"2GPU={config_2gpu.effective_batch_size:4d} {match}")
            
            assert config_1gpu.effective_batch_size == config_2gpu.effective_batch_size == target


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])


"""
End-to-End Integration Test for CBS Branching Experiments

This test runs the complete workflow:
1. Generates config files using generate_branch_configs_with_grad_accum.py
2. Verifies config files are created correctly
3. Checks config file content is valid
4. Verifies effective batch sizes match expectations

Run with: python test_end_to_end.py
"""

import os
import sys
import tempfile
import shutil
from pathlib import Path

# Add parent directory to path to import modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import auto_gradient_accumulation as aga
import generate_branch_configs_with_grad_accum as gen


def test_config_generation_end_to_end():
    """Test complete config generation workflow."""
    print("=" * 80)
    print("END-TO-END CONFIG GENERATION TEST")
    print("=" * 80)
    print()
    
    # Create temporary directory
    temp_dir = tempfile.mkdtemp()
    print(f"Using temp directory: {temp_dir}")
    print()
    
    try:
        # Setup CBS parameters
        base_batch = 32
        k_values = [1, 2, 4, 8, 16, 32]
        max_micro = 32
        num_gpus = 1  # Test for single GPU scenario
        
        print("STEP 1: Generate configs for all k-values")
        print("-" * 80)
        
        configs = {}
        for k in k_values:
            target = k * base_batch
            config = aga.calculate_gradient_accumulation(target, max_micro, num_gpus)
            gen.generate_config_file(k, temp_dir, config)
            configs[k] = config
            print(f"  k={k:2d}: Generated config for batch size {target}")
        
        print()
        
        # Verify all files exist
        print("STEP 2: Verify all config files exist")
        print("-" * 80)
        
        all_exist = True
        for k in k_values:
            config_file = os.path.join(temp_dir, f"config_branch_k{k}.py")
            if os.path.exists(config_file):
                size = os.path.getsize(config_file)
                print(f"  ✓ config_branch_k{k}.py ({size} bytes)")
            else:
                print(f"  ✗ config_branch_k{k}.py MISSING!")
                all_exist = False
        
        assert all_exist, "Some config files missing!"
        print()
        
        # Verify file contents
        print("STEP 3: Verify config file contents")
        print("-" * 80)
        
        for k in k_values:
            config_file = os.path.join(temp_dir, f"config_branch_k{k}.py")
            
            with open(config_file, 'r') as f:
                content = f.read()
            
            # Check for required fields
            config = configs[k]
            
            # Should have batch_size
            assert f"batch_size = {config.micro_batch_size_per_gpu}" in content, \
                f"k={k}: batch_size not found or incorrect"
            
            # Should have gradient_accumulation_steps
            assert f"gradient_accumulation_steps = {config.gradient_accumulation_steps}" in content, \
                f"k={k}: gradient_accumulation_steps not found or incorrect"
            
            # Should mention k value
            assert f"k={k}" in content, f"k={k}: k value not mentioned"
            
            # Should be valid Python
            try:
                compile(content, config_file, 'exec')
            except SyntaxError as e:
                raise AssertionError(f"k={k}: Config has syntax error: {e}")
            
            print(f"  ✓ config_branch_k{k}.py: Valid Python with correct values")
        
        print()
        
        # Verify effective batch sizes
        print("STEP 4: Verify effective batch sizes")
        print("-" * 80)
        
        expected_batches = {1: 32, 2: 64, 4: 128, 8: 256, 16: 512, 32: 1024}
        
        for k, expected in expected_batches.items():
            actual = configs[k].effective_batch_size
            assert actual == expected, \
                f"k={k}: effective batch {actual} != expected {expected}"
            print(f"  ✓ k={k:2d}: effective_batch_size = {actual:4d} ✓")
        
        print()
        
        # Success!
        print("=" * 80)
        print("✓✓✓ END-TO-END TEST PASSED! ✓✓✓")
        print("=" * 80)
        print()
        print("Verified:")
        print("  • All 6 config files generated successfully")
        print("  • All config files have valid Python syntax")
        print("  • All config files contain correct batch_size values")
        print("  • All config files contain correct gradient_accumulation_steps")
        print("  • All effective batch sizes match expectations (32→1024)")
        print()
        
    finally:
        # Cleanup
        shutil.rmtree(temp_dir)
        print(f"Cleaned up temp directory: {temp_dir}")


def test_multi_gpu_configs():
    """Test config generation for multi-GPU scenario."""
    print()
    print("=" * 80)
    print("MULTI-GPU CONFIG GENERATION TEST")
    print("=" * 80)
    print()
    
    temp_dir = tempfile.mkdtemp()
    
    try:
        base_batch = 32
        k_values = [1, 2, 4, 8, 16, 32]
        max_micro = 32
        num_gpus = 2  # Test for two GPU scenario
        
        print("Testing with 2 GPUs...")
        print()
        
        configs_2gpu = {}
        for k in k_values:
            target = k * base_batch
            config = aga.calculate_gradient_accumulation(target, max_micro, num_gpus)
            configs_2gpu[k] = config
        
        # Now test with 1 GPU
        configs_1gpu = {}
        for k in k_values:
            target = k * base_batch
            config = aga.calculate_gradient_accumulation(target, max_micro, 1)
            configs_1gpu[k] = config
        
        # Compare
        print("Comparing 1 GPU vs 2 GPU configs:")
        print("-" * 80)
        
        all_match = True
        for k in k_values:
            eff_1 = configs_1gpu[k].effective_batch_size
            eff_2 = configs_2gpu[k].effective_batch_size
            match = "✓" if eff_1 == eff_2 else "✗"
            
            print(f"  k={k:2d}: 1GPU={eff_1:4d}, 2GPU={eff_2:4d} {match}")
            
            if eff_1 != eff_2:
                all_match = False
        
        assert all_match, "Effective batches don't match between GPU counts!"
        
        print()
        print("✓ All effective batches match across GPU counts!")
        print()
        
    finally:
        shutil.rmtree(temp_dir)


def test_cbs_specific_values():
    """Test specific values for CBS experiments."""
    print()
    print("=" * 80)
    print("CBS-SPECIFIC VALUES TEST")
    print("=" * 80)
    print()
    
    base_batch = 32
    max_micro = 32
    
    print("Testing specific CBS configurations:")
    print("-" * 80)
    print()
    
    # Test k=1 (base case)
    print("k=1 (Base Case): batch=32")
    config_1 = aga.calculate_gradient_accumulation(32, max_micro, 1)
    print(f"  Micro batch per GPU: {config_1.micro_batch_size_per_gpu}")
    print(f"  Gradient accum steps: {config_1.gradient_accumulation_steps}")
    print(f"  Effective batch: {config_1.effective_batch_size}")
    assert config_1.micro_batch_size_per_gpu == 32
    assert config_1.gradient_accumulation_steps == 1
    assert config_1.effective_batch_size == 32
    print("  ✓ Fits perfectly, no accumulation needed")
    print()
    
    # Test k=8 (medium case)
    print("k=8 (Medium Case): batch=256")
    config_8 = aga.calculate_gradient_accumulation(256, max_micro, 1)
    print(f"  Micro batch per GPU: {config_8.micro_batch_size_per_gpu}")
    print(f"  Gradient accum steps: {config_8.gradient_accumulation_steps}")
    print(f"  Effective batch: {config_8.effective_batch_size}")
    assert config_8.effective_batch_size == 256
    print("  ✓ Uses gradient accumulation correctly")
    print()
    
    # Test k=32 (maximum case)
    print("k=32 (Maximum Case): batch=1024")
    config_32 = aga.calculate_gradient_accumulation(1024, max_micro, 1)
    print(f"  Micro batch per GPU: {config_32.micro_batch_size_per_gpu}")
    print(f"  Gradient accum steps: {config_32.gradient_accumulation_steps}")
    print(f"  Effective batch: {config_32.effective_batch_size}")
    assert config_32.effective_batch_size == 1024
    assert config_32.gradient_accumulation_steps == 32
    print("  ✓ Maximum gradient accumulation (32 steps)")
    print()
    
    print("✓ All CBS-specific values correct!")
    print()


if __name__ == "__main__":
    print()
    print("╔" + "═" * 78 + "╗")
    print("║" + " " * 78 + "║")
    print("║" + "  CBS BRANCHING EXPERIMENTS - END-TO-END INTEGRATION TESTS".center(78) + "║")
    print("║" + " " * 78 + "║")
    print("╚" + "═" * 78 + "╝")
    print()
    
    all_passed = True
    
    try:
        # Run all tests
        test_config_generation_end_to_end()
        test_multi_gpu_configs()
        test_cbs_specific_values()
        
        # Final summary
        print()
        print("╔" + "═" * 78 + "╗")
        print("║" + " " * 78 + "║")
        print("║" + "  ✓✓✓ ALL END-TO-END TESTS PASSED! ✓✓✓".center(78) + "║")
        print("║" + " " * 78 + "║")
        print("╚" + "═" * 78 + "╝")
        print()
        print("Your complete CBS branching experiments system is verified and ready!")
        print()
        print("What was tested:")
        print("  ✓ Config file generation (all 6 k-values)")
        print("  ✓ File content validity (Python syntax)")
        print("  ✓ Batch size calculations (32 → 1024)")
        print("  ✓ Gradient accumulation logic")
        print("  ✓ Single-GPU configurations")
        print("  ✓ Multi-GPU configurations")
        print("  ✓ Cross-hardware consistency")
        print("  ✓ CBS-specific values (k=1,8,32)")
        print()
        print("Next step: Share CBS_TEAM_CONFIG.md with your team!")
        print()
        
    except AssertionError as e:
        print()
        print("✗" * 40)
        print(f"TEST FAILED: {e}")
        print("✗" * 40)
        all_passed = False
        sys.exit(1)
    except Exception as e:
        print()
        print("✗" * 40)
        print(f"UNEXPECTED ERROR: {e}")
        import traceback
        traceback.print_exc()
        print("✗" * 40)
        all_passed = False
        sys.exit(1)


#!/usr/bin/env python
"""
Verify Installation and Test All Components

This script checks that all files are present and working correctly.
Run this before starting your experiments.

Usage:
    python verify_installation.py
"""

import os
import sys
import importlib.util

# Color codes for output
GREEN = '\033[92m'
RED = '\033[91m'
YELLOW = '\033[93m'
BLUE = '\033[94m'
RESET = '\033[0m'
BOLD = '\033[1m'


def print_header(text):
    """Print a section header."""
    print(f"\n{BOLD}{BLUE}{'='*80}{RESET}")
    print(f"{BOLD}{BLUE}{text}{RESET}")
    print(f"{BOLD}{BLUE}{'='*80}{RESET}\n")


def print_success(text):
    """Print success message."""
    print(f"{GREEN}✓ {text}{RESET}")


def print_error(text):
    """Print error message."""
    print(f"{RED}✗ {text}{RESET}")


def print_warning(text):
    """Print warning message."""
    print(f"{YELLOW}⚠ {text}{RESET}")


def check_file_exists(filepath, description):
    """Check if a file exists."""
    if os.path.exists(filepath):
        print_success(f"{description}: {filepath}")
        return True
    else:
        print_error(f"{description} NOT FOUND: {filepath}")
        return False


def check_import(module_name, filepath):
    """Check if a module can be imported."""
    try:
        spec = importlib.util.spec_from_file_location(module_name, filepath)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        print_success(f"Module '{module_name}' imports successfully")
        return True
    except Exception as e:
        print_error(f"Module '{module_name}' import failed: {e}")
        return False


def main():
    """Run all verification checks."""
    print(f"\n{BOLD}Branching Experiments - Installation Verification{RESET}")
    
    all_passed = True
    
    # Get parent directory (where the main scripts are located)
    parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    os.chdir(parent_dir)  # Change to parent directory for relative paths
    sys.path.insert(0, parent_dir)  # Add to path for imports
    
    # =================================================================
    # Check Original Files
    # =================================================================
    print_header("1. Checking Original Files")
    
    original_files = [
        ("generate_branch_configs.py", "Original config generator"),
        ("run_branch_sweep.py", "Sweep runner"),
        ("branching.sh", "SLURM script"),
    ]
    
    for filename, desc in original_files:
        if not check_file_exists(filename, desc):
            all_passed = False
    
    # =================================================================
    # Check New Core Files
    # =================================================================
    print_header("2. Checking New Core Files")
    
    core_files = [
        ("auto_gradient_accumulation.py", "Gradient accumulation module"),
        ("generate_branch_configs_with_grad_accum.py", "Enhanced config generator"),
        ("find_max_batch_size.py", "GPU batch size finder"),
    ]
    
    for filename, desc in core_files:
        if not check_file_exists(filename, desc):
            all_passed = False
    
    # =================================================================
    # Check Test Files
    # =================================================================
    print_header("3. Checking Test Files")
    
    test_files = [
        ("tests/test_generate_branch_configs.py", "Config generator tests"),
        ("tests/test_run_branch_sweep.py", "Sweep runner tests"),
        ("tests/test_auto_gradient_accumulation.py", "Gradient accumulation tests"),
    ]
    
    for filename, desc in test_files:
        if not check_file_exists(filename, desc):
            all_passed = False
    
    # =================================================================
    # Check Documentation
    # =================================================================
    print_header("4. Checking Documentation")
    
    doc_files = [
        ("README_TESTING_AND_GRAD_ACCUM.md", "Comprehensive guide"),
        ("QUICK_START.md", "Quick start guide"),
        ("SUMMARY.md", "Summary document"),
        ("requirements_testing.txt", "Testing dependencies"),
    ]
    
    for filename, desc in doc_files:
        if not check_file_exists(filename, desc):
            all_passed = False
    
    # =================================================================
    # Check Imports
    # =================================================================
    print_header("5. Checking Module Imports")
    
    if os.path.exists("auto_gradient_accumulation.py"):
        if not check_import("auto_gradient_accumulation", "auto_gradient_accumulation.py"):
            all_passed = False
    
    if os.path.exists("generate_branch_configs.py"):
        if not check_import("generate_branch_configs", "generate_branch_configs.py"):
            all_passed = False
    
    # =================================================================
    # Check Dependencies
    # =================================================================
    print_header("6. Checking Python Dependencies")
    
    required_modules = [
        ("math", "Standard library"),
        ("os", "Standard library"),
        ("subprocess", "Standard library"),
        ("json", "Standard library"),
    ]
    
    for module_name, source in required_modules:
        try:
            __import__(module_name)
            print_success(f"Module '{module_name}' available ({source})")
        except ImportError:
            print_error(f"Module '{module_name}' not available")
            all_passed = False
    
    # Optional dependencies
    print("\nOptional dependencies:")
    
    optional_modules = [
        ("pytest", "For running tests"),
        ("torch", "For GPU batch size testing"),
    ]
    
    for module_name, purpose in optional_modules:
        try:
            __import__(module_name)
            print_success(f"Module '{module_name}' available ({purpose})")
        except ImportError:
            print_warning(f"Module '{module_name}' not available ({purpose})")
            print(f"          Install with: pip install {module_name}")
    
    # =================================================================
    # Test Gradient Accumulation Logic
    # =================================================================
    print_header("7. Testing Gradient Accumulation Logic")
    
    if os.path.exists("auto_gradient_accumulation.py"):
        try:
            import auto_gradient_accumulation as aga
            
            # Test basic calculation
            config = aga.calculate_gradient_accumulation(64, 16, num_gpus=1)
            assert config.micro_batch_size_per_gpu == 16
            assert config.gradient_accumulation_steps == 4
            assert config.num_gpus == 1
            assert config.effective_batch_size == 64
            print_success("Basic gradient accumulation calculation works")
            
            # Test memory estimation
            max_batch = aga.estimate_max_batch_size_from_memory(
                model_params=45_000_000,
                sequence_length=128,
                gpu_memory_gb=24
            )
            assert max_batch > 0
            print_success(f"Memory estimation works (estimated max batch: {max_batch})")
            
            # Test multi-GPU calculation
            config_multigpu = aga.calculate_gradient_accumulation(128, 32, num_gpus=4)
            assert config_multigpu.num_gpus == 4
            assert config_multigpu.effective_batch_size == 128
            print_success(f"Multi-GPU calculation works ({config_multigpu.micro_batch_size_per_gpu}×{config_multigpu.num_gpus}×{config_multigpu.gradient_accumulation_steps}={config_multigpu.effective_batch_size})")
            
            # Test GPU detection
            num_gpus = aga.get_num_gpus()
            print_success(f"GPU detection works (detected {num_gpus} GPU(s))")
            
        except Exception as e:
            print_error(f"Gradient accumulation test failed: {e}")
            all_passed = False
    
    # =================================================================
    # Summary
    # =================================================================
    print_header("Summary")
    
    if all_passed:
        print(f"{GREEN}{BOLD}✓ All checks passed!{RESET}")
        print(f"\n{GREEN}Your installation is ready to use.{RESET}\n")
        print("Next steps:")
        print("  1. Run: python auto_gradient_accumulation.py")
        print("  2. If you have a GPU: python find_max_batch_size.py")
        print("  3. Edit and run: python generate_branch_configs_with_grad_accum.py")
        print("  4. Start experiments: python run_branch_sweep.py")
        print("\nFor more information, see:")
        print("  - QUICK_START.md (5-minute guide)")
        print("  - README_TESTING_AND_GRAD_ACCUM.md (comprehensive guide)")
        print("  - SUMMARY.md (overview)")
        return 0
    else:
        print(f"{RED}{BOLD}✗ Some checks failed.{RESET}")
        print(f"\n{RED}Please fix the issues above before proceeding.{RESET}\n")
        return 1


if __name__ == "__main__":
    sys.exit(main())


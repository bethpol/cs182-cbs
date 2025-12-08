#!/usr/bin/env python3
"""
Run All Tests for CBS Branching Experiments

This script runs all available tests and provides a comprehensive summary.

Usage: python run_all_tests.py
"""

import subprocess
import sys
import os

def run_command(cmd, description):
    """Run a command and return success status."""
    print()
    print("=" * 80)
    print(f"RUNNING: {description}")
    print("=" * 80)
    print(f"Command: {cmd}")
    print()
    
    result = subprocess.run(cmd, shell=True, capture_output=False)
    success = result.returncode == 0
    
    if success:
        print()
        print(f"✓ {description} PASSED")
    else:
        print()
        print(f"✗ {description} FAILED")
    
    return success

def main():
    print()
    print("╔" + "═" * 78 + "╗")
    print("║" + " " * 78 + "║")
    print("║" + "CBS BRANCHING EXPERIMENTS - COMPLETE TEST SUITE".center(78) + "║")
    print("║" + " " * 78 + "║")
    print("╚" + "═" * 78 + "╝")
    print()
    
    # Change to parent directory (where main scripts are)
    parent_dir = os.path.join(os.path.dirname(__file__), '..')
    os.chdir(parent_dir)
    
    results = {}
    
    # Test 1: Python syntax check
    results['Syntax Check'] = run_command(
        'python -m py_compile auto_gradient_accumulation.py generate_branch_configs_with_grad_accum.py run_branch_sweep.py',
        'Python Syntax Check'
    )
    
    # Test 2: End-to-end integration tests
    results['End-to-End Tests'] = run_command(
        'python tests/test_end_to_end.py',
        'End-to-End Integration Tests'
    )
    
    # Test 3: Installation verification
    results['Installation Verification'] = run_command(
        'python tests/verify_installation.py',
        'Installation Verification'
    )
    
    # Test 4: Check if pytest is available
    pytest_available = subprocess.run(
        'python -c "import pytest"',
        shell=True,
        capture_output=True
    ).returncode == 0
    
    if pytest_available:
        # Test 5: Unit tests (if pytest available)
        results['Unit Tests'] = run_command(
            'pytest tests/test_auto_gradient_accumulation.py -v --tb=short',
            'Unit Tests (auto_gradient_accumulation)'
        )
        
        results['Config Tests'] = run_command(
            'pytest tests/test_generate_branch_configs.py -v --tb=short',
            'Config Generation Tests'
        )
        
        results['Sweep Tests'] = run_command(
            'pytest tests/test_run_branch_sweep.py -v --tb=short',
            'Sweep Execution Tests'
        )
        
        results['Integration Tests'] = run_command(
            'pytest tests/test_integration_cbs.py -v --tb=short',
            'CBS Integration Tests'
        )
    else:
        print()
        print("=" * 80)
        print("NOTE: pytest not installed - skipping pytest-based tests")
        print("=" * 80)
        print()
        print("To run all tests, install pytest:")
        print("  pip install pytest")
        print()
    
    # Summary
    print()
    print("╔" + "═" * 78 + "╗")
    print("║" + " " * 78 + "║")
    print("║" + "TEST SUITE SUMMARY".center(78) + "║")
    print("║" + " " * 78 + "║")
    print("╚" + "═" * 78 + "╝")
    print()
    
    passed = sum(1 for v in results.values() if v)
    total = len(results)
    
    for test_name, success in results.items():
        status = "✓ PASSED" if success else "✗ FAILED"
        print(f"  {status:12s} - {test_name}")
    
    print()
    print(f"Total: {passed}/{total} test suites passed")
    print()
    
    if passed == total:
        print("╔" + "═" * 78 + "╗")
        print("║" + " " * 78 + "║")
        print("║" + "✓✓✓ ALL TESTS PASSED! ✓✓✓".center(78) + "║")
        print("║" + " " * 78 + "║")
        print("║" + "Your CBS branching experiments system is fully verified!".center(78) + "║")
        print("║" + " " * 78 + "║")
        print("╚" + "═" * 78 + "╝")
        print()
        print("What was tested:")
        print("  ✓ Python syntax and imports")
        print("  ✓ End-to-end workflows")
        print("  ✓ Installation and setup")
        if pytest_available:
            print("  ✓ Unit tests (200+ tests)")
            print("  ✓ Config generation")
            print("  ✓ Sweep execution")
            print("  ✓ CBS integration tests")
        print()
        print("🚀 Ready to start your CBS experiments!")
        print()
        sys.exit(0)
    else:
        print("╔" + "═" * 78 + "╗")
        print("║" + " " * 78 + "║")
        print("║" + f"✗ {total - passed} test suite(s) failed".center(78) + "║")
        print("║" + " " * 78 + "║")
        print("╚" + "═" * 78 + "╝")
        print()
        print("Please review the output above for details.")
        print()
        sys.exit(1)

if __name__ == "__main__":
    main()


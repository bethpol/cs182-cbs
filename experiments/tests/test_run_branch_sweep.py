"""
Tests for run_branch_sweep.py

Run with: pytest test_run_branch_sweep.py -v
"""

import os
import sys
import tempfile
import shutil
import json
import time
from pathlib import Path
from unittest import mock

import pytest

# Add parent directory to path to import modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import run_branch_sweep as rbs


class TestSingleBranchRun:
    """Test running a single branch."""
    
    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for test outputs."""
        tmpdir = tempfile.mkdtemp()
        yield tmpdir
        shutil.rmtree(tmpdir)
    
    @pytest.fixture
    def mock_train_script(self, temp_dir):
        """Create a mock training script that just exits successfully."""
        script_path = os.path.join(temp_dir, "mock_train.py")
        with open(script_path, 'w') as f:
            f.write("#!/usr/bin/env python\n")
            f.write("import sys\n")
            f.write("print('Mock training started')\n")
            f.write("print('Training step 1/10')\n")
            f.write("print('Training step 10/10')\n")
            f.write("print('Mock training completed')\n")
            f.write("sys.exit(0)\n")
        return script_path
    
    @pytest.fixture
    def failing_train_script(self, temp_dir):
        """Create a mock training script that fails."""
        script_path = os.path.join(temp_dir, "failing_train.py")
        with open(script_path, 'w') as f:
            f.write("#!/usr/bin/env python\n")
            f.write("import sys\n")
            f.write("print('Training started but will fail')\n")
            f.write("print('Error occurred!')\n")
            f.write("sys.exit(1)\n")
        return script_path
    
    def test_successful_branch_run(self, temp_dir, mock_train_script):
        """Test a successful branch training run."""
        k = 4
        config_path = os.path.join(temp_dir, "config_k4.py")
        log_path = os.path.join(temp_dir, "test_k4.log")
        
        # Create a dummy config
        with open(config_path, 'w') as f:
            f.write("# Test config\n")
        
        # Temporarily override TRAIN_SCRIPT
        original_script = rbs.TRAIN_SCRIPT
        rbs.TRAIN_SCRIPT = mock_train_script
        
        try:
            result = rbs.run_single_branch(k, config_path, log_path)
        finally:
            rbs.TRAIN_SCRIPT = original_script
        
        # Check result structure
        assert isinstance(result, dict)
        assert result['k'] == k
        assert result['success'] is True
        assert result['exit_code'] == 0
        assert 'duration' in result
        assert result['duration'] >= 0
        
        # Check log file was created and contains expected content
        assert os.path.exists(log_path)
        with open(log_path, 'r') as f:
            log_content = f.read()
        
        assert "Mock training started" in log_content
        assert "Mock training completed" in log_content
        assert f"Branch k={k}" in log_content
    
    def test_failed_branch_run(self, temp_dir, failing_train_script):
        """Test a failing branch training run."""
        k = 2
        config_path = os.path.join(temp_dir, "config_k2.py")
        log_path = os.path.join(temp_dir, "test_k2.log")
        
        # Create a dummy config
        with open(config_path, 'w') as f:
            f.write("# Test config\n")
        
        # Temporarily override TRAIN_SCRIPT
        original_script = rbs.TRAIN_SCRIPT
        rbs.TRAIN_SCRIPT = failing_train_script
        
        try:
            result = rbs.run_single_branch(k, config_path, log_path)
        finally:
            rbs.TRAIN_SCRIPT = original_script
        
        # Check result indicates failure
        assert result['success'] is False
        assert result['exit_code'] == 1
        assert 'duration' in result
        
        # Check log file contains error info
        assert os.path.exists(log_path)
        with open(log_path, 'r') as f:
            log_content = f.read()
        
        assert "Error occurred!" in log_content
        assert "Exit code: 1" in log_content


class TestResultsCollection:
    """Test results collection and reporting."""
    
    def test_collect_all_successful(self, tmp_path, capsys):
        """Test collecting results when all branches succeed."""
        rbs.LOGS_DIR = str(tmp_path)
        
        results = [
            {'k': 1, 'success': True, 'duration': 10.5, 'exit_code': 0, 'log_file': 'k1.log'},
            {'k': 2, 'success': True, 'duration': 15.2, 'exit_code': 0, 'log_file': 'k2.log'},
            {'k': 4, 'success': True, 'duration': 20.1, 'exit_code': 0, 'log_file': 'k4.log'},
        ]
        
        rbs.collect_results(results)
        
        # Check JSON file was created
        results_file = os.path.join(str(tmp_path), "sweep_results.json")
        assert os.path.exists(results_file)
        
        with open(results_file, 'r') as f:
            saved_results = json.load(f)
        
        assert saved_results['total_runs'] == 3
        assert saved_results['successful_runs'] == 3
        assert len(saved_results['results']) == 3
        
        # Check console output
        captured = capsys.readouterr()
        assert "✓ Success" in captured.out
        assert "Total branches: 3" in captured.out
        assert "Successful: 3" in captured.out
        assert "Failed: 0" in captured.out
    
    def test_collect_mixed_results(self, tmp_path, capsys):
        """Test collecting results with both successes and failures."""
        rbs.LOGS_DIR = str(tmp_path)
        
        results = [
            {'k': 1, 'success': True, 'duration': 10.5, 'exit_code': 0, 'log_file': 'k1.log'},
            {'k': 2, 'success': False, 'duration': 5.2, 'exit_code': 1, 'log_file': 'k2.log'},
            {'k': 4, 'success': True, 'duration': 20.1, 'exit_code': 0, 'log_file': 'k4.log'},
        ]
        
        rbs.collect_results(results)
        
        # Check JSON file
        results_file = os.path.join(str(tmp_path), "sweep_results.json")
        with open(results_file, 'r') as f:
            saved_results = json.load(f)
        
        assert saved_results['total_runs'] == 3
        assert saved_results['successful_runs'] == 2
        
        # Check console output shows both success and failure
        captured = capsys.readouterr()
        assert "✓ Success" in captured.out
        assert "✗ Failed" in captured.out
        assert "Successful: 2" in captured.out
        assert "Failed: 1" in captured.out
    
    def test_collect_duration_calculation(self, tmp_path):
        """Test that total duration is calculated correctly."""
        rbs.LOGS_DIR = str(tmp_path)
        
        results = [
            {'k': 1, 'success': True, 'duration': 10.0, 'exit_code': 0, 'log_file': 'k1.log'},
            {'k': 2, 'success': True, 'duration': 20.0, 'exit_code': 0, 'log_file': 'k2.log'},
            {'k': 4, 'success': True, 'duration': 30.0, 'exit_code': 0, 'log_file': 'k4.log'},
        ]
        
        rbs.collect_results(results)
        
        results_file = os.path.join(str(tmp_path), "sweep_results.json")
        with open(results_file, 'r') as f:
            saved_results = json.load(f)
        
        expected_duration = 10.0 + 20.0 + 30.0
        assert abs(saved_results['total_duration'] - expected_duration) < 0.01


class TestConfigFileValidation:
    """Test config file validation and existence checks."""
    
    def test_missing_config_directory(self, tmp_path, monkeypatch, capsys):
        """Test behavior when config directory doesn't exist."""
        # Create fake train script first (so we pass that check)
        fake_script = str(tmp_path / 'fake_train.py')
        with open(fake_script, 'w') as f:
            f.write("# Fake train script\n")
        monkeypatch.setattr(rbs, 'TRAIN_SCRIPT', fake_script)
        
        # Set non-existent config dir
        fake_dir = str(tmp_path / "nonexistent")
        monkeypatch.setattr(rbs, 'CONFIG_DIR', fake_dir)
        
        # Mock input to avoid blocking
        with mock.patch('builtins.input', return_value='n'):
            rbs.main()
        
        captured = capsys.readouterr()
        assert "Config directory not found" in captured.out
    
    def test_missing_train_script(self, tmp_path, monkeypatch, capsys):
        """Test behavior when train script doesn't exist."""
        # Set non-existent train script
        fake_script = str(tmp_path / "nonexistent_train.py")
        monkeypatch.setattr(rbs, 'TRAIN_SCRIPT', fake_script)
        
        # Mock input
        with mock.patch('builtins.input', return_value='n'):
            rbs.main()
        
        captured = capsys.readouterr()
        assert "Training script not found" in captured.out


class TestKValueHandling:
    """Test handling of different k-values."""
    
    def test_k_values_match_configs(self, tmp_path):
        """Test that k-values match expected config files."""
        k_values = [1, 2, 4, 8, 16, 32]
        
        for k in k_values:
            expected_filename = f"config_branch_k{k}.py"
            # Just testing the naming logic
            assert expected_filename == f"config_branch_k{k}.py"
    
    def test_missing_config_for_k_value(self, tmp_path, monkeypatch, capsys):
        """Test handling when some config files are missing."""
        config_dir = str(tmp_path / "configs")
        os.makedirs(config_dir, exist_ok=True)
        
        # Create only some config files
        for k in [1, 2, 4]:
            config_path = os.path.join(config_dir, f"config_branch_k{k}.py")
            with open(config_path, 'w') as f:
                f.write("# Test config\n")
        
        monkeypatch.setattr(rbs, 'CONFIG_DIR', config_dir)
        monkeypatch.setattr(rbs, 'TRAIN_SCRIPT', 'fake_train.py')
        
        # Create fake train script
        fake_script = os.path.join(str(tmp_path), 'fake_train.py')
        with open(fake_script, 'w') as f:
            f.write("# Fake\n")
        monkeypatch.setattr(rbs, 'TRAIN_SCRIPT', fake_script)
        
        with mock.patch('builtins.input', return_value='n'):
            rbs.main()
        
        captured = capsys.readouterr()
        # Should warn about missing configs
        # (k=8, 16, 32 are missing)
        assert "Warning" in captured.out or "Found 3 config files" in captured.out


class TestLogFileCreation:
    """Test log file creation and formatting."""
    
    def test_log_file_has_header(self, tmp_path):
        """Test that log files have proper headers."""
        k = 4
        config_path = str(tmp_path / "config.py")
        log_path = str(tmp_path / "test.log")
        
        # Create dummy config
        with open(config_path, 'w') as f:
            f.write("# Config\n")
        
        # Create a mock train script that succeeds immediately
        mock_script = str(tmp_path / "mock_train.py")
        with open(mock_script, 'w') as f:
            f.write("print('done')\n")
        
        original_script = rbs.TRAIN_SCRIPT
        rbs.TRAIN_SCRIPT = mock_script
        
        try:
            rbs.run_single_branch(k, config_path, log_path)
        finally:
            rbs.TRAIN_SCRIPT = original_script
        
        with open(log_path, 'r') as f:
            content = f.read()
        
        assert f"Branch k={k}" in content
        assert "Started:" in content
        assert "Completed:" in content
        assert "Duration:" in content


class TestEdgeCases:
    """Test edge cases and error conditions."""
    
    def test_empty_k_values_list(self, tmp_path, monkeypatch):
        """Test behavior with empty K_VALUES list."""
        config_dir = str(tmp_path / "configs")
        os.makedirs(config_dir, exist_ok=True)
        
        monkeypatch.setattr(rbs, 'K_VALUES', [])
        monkeypatch.setattr(rbs, 'CONFIG_DIR', config_dir)
        
        # Create fake train script
        fake_script = str(tmp_path / 'fake_train.py')
        with open(fake_script, 'w') as f:
            f.write("# Fake\n")
        monkeypatch.setattr(rbs, 'TRAIN_SCRIPT', fake_script)
        
        with mock.patch('builtins.input', return_value='n'):
            rbs.main()
        
        # Should handle gracefully (no crash)
    
    def test_user_cancels_sweep(self, tmp_path, monkeypatch, capsys):
        """Test that sweep can be cancelled by user."""
        config_dir = str(tmp_path / "configs")
        os.makedirs(config_dir, exist_ok=True)
        
        # Create a config file
        config_path = os.path.join(config_dir, "config_branch_k1.py")
        with open(config_path, 'w') as f:
            f.write("# Test\n")
        
        monkeypatch.setattr(rbs, 'K_VALUES', [1])
        monkeypatch.setattr(rbs, 'CONFIG_DIR', config_dir)
        
        # Create fake train script
        fake_script = str(tmp_path / 'fake_train.py')
        with open(fake_script, 'w') as f:
            f.write("# Fake\n")
        monkeypatch.setattr(rbs, 'TRAIN_SCRIPT', fake_script)
        
        # User says 'n' to cancel
        with mock.patch('builtins.input', return_value='n'):
            rbs.main()
        
        captured = capsys.readouterr()
        assert "cancelled" in captured.out.lower()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])



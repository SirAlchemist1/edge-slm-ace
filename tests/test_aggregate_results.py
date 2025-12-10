"""Tests for aggregate_results.py functionality."""

import json
import sys
import tempfile
from pathlib import Path

import pandas as pd
import pytest

# Add scripts to path for testing
sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

from aggregate_results import (
    aggregate_metrics,
    derive_run_name_from_path,
    extract_device_from_path,
    find_metrics_files,
    load_and_extract_metrics,
)


class TestAggregateResults:
    """Tests for aggregate_results functionality."""
    
    def test_find_metrics_files(self):
        """Test finding metrics.json files recursively."""
        with tempfile.TemporaryDirectory() as tmpdir:
            results_root = Path(tmpdir)
            
            # Create directory structure
            (results_root / "model1" / "task1" / "baseline" / "cpu").mkdir(parents=True)
            (results_root / "model1" / "task1" / "ace_full" / "cpu").mkdir(parents=True)
            
            # Create metrics files
            metrics1 = results_root / "model1" / "task1" / "baseline" / "cpu" / "metrics.json"
            metrics2 = results_root / "model1" / "task1" / "ace_full" / "cpu" / "metrics.json"
            
            metrics1.write_text(json.dumps({"model_id": "model1", "task_name": "task1", "mode": "baseline", "accuracy": 0.5}))
            metrics2.write_text(json.dumps({"model_id": "model1", "task_name": "task1", "mode": "ace_full", "accuracy": 0.7}))
            
            found = find_metrics_files(results_root)
            
            assert len(found) == 2
            assert metrics1 in found
            assert metrics2 in found
    
    def test_extract_device_from_path(self):
        """Test extracting device from path."""
        path1 = Path("results/model1/task1/baseline/cpu/metrics.json")
        path2 = Path("results/model1/task1/baseline/cuda/metrics.json")
        path3 = Path("results/model1/task1/baseline/metrics.json")
        
        assert extract_device_from_path(path1) == "cpu"
        assert extract_device_from_path(path2) == "cuda"
        assert extract_device_from_path(path3) is None
    
    def test_derive_run_name_from_path(self):
        """Test deriving run name from path."""
        path = Path("results/model1/task1/baseline/cpu/metrics.json")
        run_name = derive_run_name_from_path(path)
        
        assert "model1" in run_name
        assert "task1" in run_name
        assert "baseline" in run_name
    
    def test_load_and_extract_metrics(self):
        """Test loading and extracting metrics from JSON."""
        with tempfile.TemporaryDirectory() as tmpdir:
            metrics_path = Path(tmpdir) / "metrics.json"
            
            metrics_data = {
                "run_name": "test_run",
                "model_id": "sshleifer/tiny-gpt2",
                "task_name": "tatqa_tiny",
                "domain": "finance",
                "mode": "baseline",
                "device_used": "cpu",
                "accuracy": 0.5,
                "avg_latency_ms": 100.0,
                "num_examples": 10,
                "playbook": {
                    "initial_size": 0,
                    "final_size": 5,
                    "entries_added": 5,
                },
            }
            
            metrics_path.write_text(json.dumps(metrics_data))
            
            row = load_and_extract_metrics(metrics_path)
            
            assert row is not None
            assert row["run_name"] == "test_run"
            assert row["model_id"] == "sshleifer/tiny-gpt2"
            assert row["task_name"] == "tatqa_tiny"
            assert row["mode"] == "baseline"
            assert row["accuracy"] == 0.5
            assert row["playbook_initial_size"] == 0
            assert row["playbook_final_size"] == 5
            assert row["playbook_entries_added"] == 5
    
    def test_load_and_extract_metrics_ace_mode(self):
        """Test loading metrics with ACE mode."""
        with tempfile.TemporaryDirectory() as tmpdir:
            metrics_path = Path(tmpdir) / "metrics.json"
            
            metrics_data = {
                "model_id": "sshleifer/tiny-gpt2",
                "task_name": "tatqa_tiny",
                "mode": "ace",
                "ace_mode": "ace_working_memory",
                "device_used": "cpu",
                "accuracy": 0.6,
                "playbook": {
                    "initial_size": 0,
                    "final_size": 3,
                    "entries_added": 3,
                },
            }
            
            metrics_path.write_text(json.dumps(metrics_data))
            
            row = load_and_extract_metrics(metrics_path)
            
            assert row is not None
            assert row["mode"] == "ace"
            assert row["ace_mode"] == "ace_working_memory"
            assert row["playbook_final_size"] == 3
    
    def test_aggregate_metrics(self):
        """Test aggregating multiple metrics files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            results_root = Path(tmpdir)
            
            # Create directory structure
            (results_root / "model1" / "task1" / "baseline" / "cpu").mkdir(parents=True)
            (results_root / "model1" / "task1" / "ace_full" / "cpu").mkdir(parents=True)
            (results_root / "model2" / "task2" / "baseline" / "cpu").mkdir(parents=True)
            
            # Create metrics files
            metrics1 = results_root / "model1" / "task1" / "baseline" / "cpu" / "metrics.json"
            metrics2 = results_root / "model1" / "task1" / "ace_full" / "cpu" / "metrics.json"
            metrics3 = results_root / "model2" / "task2" / "baseline" / "cpu" / "metrics.json"
            
            metrics1.write_text(json.dumps({
                "model_id": "model1",
                "task_name": "task1",
                "mode": "baseline",
                "device_used": "cpu",
                "accuracy": 0.5,
                "avg_latency_ms": 100.0,
            }))
            
            metrics2.write_text(json.dumps({
                "model_id": "model1",
                "task_name": "task1",
                "mode": "ace",
                "ace_mode": "ace_full",
                "device_used": "cpu",
                "accuracy": 0.7,
                "avg_latency_ms": 150.0,
                "playbook": {
                    "initial_size": 0,
                    "final_size": 5,
                    "entries_added": 5,
                },
            }))
            
            metrics3.write_text(json.dumps({
                "model_id": "model2",
                "task_name": "task2",
                "mode": "baseline",
                "device_used": "cpu",
                "accuracy": 0.6,
                "avg_latency_ms": 120.0,
            }))
            
            df = aggregate_metrics(results_root)
            
            assert len(df) == 3
            assert "model_id" in df.columns
            assert "task_name" in df.columns
            assert "mode" in df.columns
            assert "accuracy" in df.columns
            
            # Check specific values
            baseline_row = df[(df["model_id"] == "model1") & (df["mode"] == "baseline")]
            assert len(baseline_row) == 1
            assert baseline_row.iloc[0]["accuracy"] == 0.5
            
            ace_row = df[(df["model_id"] == "model1") & (df["ace_mode"] == "ace_full")]
            assert len(ace_row) == 1
            assert ace_row.iloc[0]["accuracy"] == 0.7
            assert ace_row.iloc[0]["playbook_final_size"] == 5
    
    def test_aggregate_metrics_empty_directory(self):
        """Test aggregating from empty directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            results_root = Path(tmpdir)
            df = aggregate_metrics(results_root)
            
            assert len(df) == 0
    
    def test_aggregate_metrics_malformed_file(self):
        """Test handling of malformed JSON files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            results_root = Path(tmpdir)
            (results_root / "model1" / "task1" / "baseline" / "cpu").mkdir(parents=True)
            
            # Create valid and invalid files
            valid_metrics = results_root / "model1" / "task1" / "baseline" / "cpu" / "metrics.json"
            invalid_metrics = results_root / "model1" / "task1" / "baseline" / "cpu" / "bad_metrics.json"
            
            valid_metrics.write_text(json.dumps({
                "model_id": "model1",
                "task_name": "task1",
                "mode": "baseline",
                "accuracy": 0.5,
            }))
            
            invalid_metrics.write_text("{ invalid json }")
            
            # Should only find valid metrics.json
            df = aggregate_metrics(results_root)
            
            assert len(df) == 1
            assert df.iloc[0]["model_id"] == "model1"

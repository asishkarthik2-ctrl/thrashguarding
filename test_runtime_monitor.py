import unittest
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd
import app


class LiveRiskTests(unittest.TestCase):
    def setUp(self):
        self.snapshot = {
            "telemetry_status": "filtered",
            "memory_percent": 75.0,
            "free_memory_percent": 25.0,
            "swap_percent": 12.0,
            "page_faults_per_sec": 120.0,
            "hard_faults_per_sec": 4.0,
            "process_count": 2,
            "visible_process_count": 2,
            "visible_process_memory_mb": 750.0,
            "process_cpu_percent": 0.0,
            "processes_started": [],
            "processes_stopped": [],
            "timestamp": "2026-10-14 12:00:00",
            "processes": [
                {"pid": 100, "name": "app-one", "resident_ratio": 0.6, "rss_mb": 500.0},
                {"pid": 200, "name": "app-two", "resident_ratio": 0.4, "rss_mb": 250.0},
            ],
        }

    def test_live_input_matches_training_feature_schema(self):
        features = app.build_live_model_input(self.snapshot)

        self.assertEqual(list(features.columns), app.LIVE_MODEL_FEATURES)
        self.assertEqual(len(features), 2)
        self.assertEqual(features.iloc[0].to_dict(), {
            "resident_ratio": 0.6,
            "memory_pressure": 0.75,
            "free_memory_percent": 0.25,
        })

    def test_model_trains_and_saves_locally_when_artifact_is_missing(self):
        training_data = pd.DataFrame({
            "resident_pages": [10, 20, 80, 90],
            "working_set_size": [100, 100, 100, 100],
            "memory_pressure": [20, 30, 80, 90],
            "free_frames": [180, 150, 20, 5],
            "page_fault": [1, 1, 0, 0],
        })
        with tempfile.TemporaryDirectory() as temp_dir:
            model_path = Path(temp_dir) / "model.pkl"
            dataset_path = Path(temp_dir) / "dataset.csv"
            training_data.to_csv(dataset_path, index=False)

            local_model, status = app.load_or_train_model(model_path, dataset_path)

            self.assertEqual(status, "trained_offline")
            self.assertTrue(model_path.is_file())
            self.assertEqual(local_model.predict_proba(
                app.prepare_training_features(training_data)
            ).shape, (4, 2))

    def test_windows_memory_counters_are_collected_in_one_command(self):
        completed_process = Mock(
            returncode=0,
            stdout=(
                r"\\laptop\memory\page faults/sec=123.4" + "\n"
                + r"\\laptop\memory\pages/sec=5.6" + "\n"
            ),
        )
        with patch.object(app.os, "name", "nt"), patch.object(
            app.subprocess, "run", return_value=completed_process
        ) as run:
            counters = app.collect_windows_memory_counters()

        self.assertEqual(counters, (123.4, 5.6))
        run.assert_called_once()

    def test_single_sample_window_does_not_smooth_live_values(self):
        collector = app.TelemetryCollector(average_window=1)
        collector._apply_moving_average({"cpu_percent": 10.0})

        latest = collector._apply_moving_average({"cpu_percent": 90.0})

        self.assertEqual(latest["cpu_percent"], 90.0)

    def test_process_collection_includes_all_readable_processes_and_cpu(self):
        processes = []
        for pid in range(1, 16):
            process = Mock()
            process.info = {
                "pid": pid,
                "name": f"app-{pid}",
                "memory_info": SimpleNamespace(
                    rss=pid * 1024 * 1024,
                    vms=pid * 2048 * 1024,
                ),
            }
            process.cpu_percent.return_value = float(pid)
            processes.append(process)

        with patch.object(app.psutil, "cpu_count", return_value=2), patch.object(
            app.psutil, "process_iter", return_value=processes
        ):
            collected = app.collect_process_memory()

        self.assertEqual(len(collected), 15)
        self.assertEqual(collected[0]["name"], "app-15")
        self.assertEqual(collected[0]["cpu_percent"], 7.5)

    def test_process_changes_are_reported_between_samples(self):
        collector = app.TelemetryCollector()
        first_sample = {
            "telemetry_status": "live",
            "process_count": 2,
            "processes": [
                {"pid": 1, "name": "existing", "rss_mb": 10.0},
                {"pid": 2, "name": "closed", "rss_mb": 10.0},
            ],
        }
        second_sample = {
            "telemetry_status": "live",
            "process_count": 2,
            "processes": [
                {"pid": 1, "name": "existing", "rss_mb": 10.0},
                {"pid": 3, "name": "opened", "rss_mb": 10.0},
            ],
        }
        with patch.object(app.psutil, "virtual_memory", return_value=SimpleNamespace(
            total=100, available=50, percent=50
        )), patch.object(app.psutil, "swap_memory", return_value=SimpleNamespace(percent=0)), patch.object(
            app.psutil, "disk_usage", return_value=SimpleNamespace(percent=0)
        ), patch.object(app.psutil, "cpu_percent", return_value=10), patch.object(
            app.psutil, "pids", return_value=[1, 2]
        ), patch.object(app, "collect_process_memory", side_effect=[
            first_sample["processes"], second_sample["processes"]
        ]), patch.object(app, "windows_has_admin_rights", return_value=False), patch.object(
            app.psutil, "Process", return_value=Mock(memory_info=Mock(return_value=SimpleNamespace(rss=1024)))
        ):
            first = collector._read_raw_snapshot()
            second = collector._read_raw_snapshot()

        self.assertEqual(first["processes_started"], [])
        self.assertEqual(second["processes_started"], ["opened"])
        self.assertEqual(second["processes_stopped"], ["closed"])

    def test_process_scores_use_model_probabilities(self):
        fake_model = Mock()
        fake_model.predict_proba.return_value = np.array([[0.2, 0.8], [0.7, 0.3]])
        with patch.object(app, "model", fake_model):
            scored = app.score_live_processes(self.snapshot)

        self.assertEqual(scored[0]["process_name"], "app-one")
        self.assertEqual(scored[0]["risk_score"], 80.0)
        self.assertEqual(scored[1]["risk_score"], 30.0)

    def test_token_booth_is_risk_based_and_advisory(self):
        predictions = [
            {"process_name": "critical", "pid": 1, "rss_mb": 100, "risk_score": 95, "risk_level": "high"},
            {"process_name": "moderate", "pid": 2, "rss_mb": 50, "risk_score": 60, "risk_level": "medium"},
            {"process_name": "low", "pid": 3, "rss_mb": 25, "risk_score": 25, "risk_level": "low"},
        ]

        booth = app.build_token_booth(predictions)

        self.assertEqual(booth["tokens_per_process"], 10)
        self.assertEqual(booth["mode"], "read-only-simulation")
        self.assertFalse(booth["controls_system_processes"])
        self.assertEqual(booth["priority_rule"], "lowest_predicted_risk_first")
        self.assertEqual([process["process_name"] for process in booth["processes"]], ["low", "moderate", "critical"])
        self.assertEqual([process["priority_rank"] for process in booth["processes"]], [1, 2, 3])
        self.assertEqual(booth["processes"][0]["refill_mode"], "steady")
        self.assertEqual(booth["processes"][1]["refill_mode"], "moderate")
        self.assertEqual(booth["processes"][2]["refill_mode"], "paused")

    def test_predict_returns_live_scores_and_os_counters(self):
        fake_model = Mock()
        fake_model.predict_proba.return_value = np.array([[0.2, 0.8], [0.7, 0.3]])
        with patch.object(app, "model", fake_model), patch.object(app, "safe_system_snapshot", return_value=self.snapshot):
            response = app.app.test_client().get("/predict")

        self.assertEqual(response.status_code, 200)
        result = response.get_json()
        self.assertEqual(result["risk_score"], 80.0)
        self.assertEqual(result["execution_mode"], "offline")
        self.assertEqual(result["model_status"], app.MODEL_STATUS)
        self.assertEqual(result["snapshot"]["hard_faults_per_sec"], 4.0)
        self.assertEqual(result["snapshot"]["visible_process_count"], 2)
        self.assertEqual(len(result["process_predictions"]), 2)
        self.assertEqual(result["snapshot"]["process_cpu_percent"], 0.0)
        self.assertNotIn("page_heat_scores", result)


if __name__ == "__main__":
    unittest.main()

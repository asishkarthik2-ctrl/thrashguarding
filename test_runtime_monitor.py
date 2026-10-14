import unittest
from unittest.mock import Mock, patch

import numpy as np
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
        self.assertEqual(result["snapshot"]["hard_faults_per_sec"], 4.0)
        self.assertEqual(len(result["process_predictions"]), 2)
        self.assertNotIn("page_heat_scores", result)


if __name__ == "__main__":
    unittest.main()

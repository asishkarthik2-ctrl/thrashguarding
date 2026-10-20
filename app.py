import os
import subprocess
import threading
import time
from collections import deque
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
import psutil
from flask import Flask, jsonify, render_template
from sklearn.ensemble import RandomForestClassifier

app = Flask(__name__)

APP_DIR = Path(__file__).resolve().parent
MODEL_PATH = APP_DIR / "page_fault_model.pkl"
DATASET_PATH = APP_DIR / "page_fault_dataset.csv"
LIVE_MODEL_FEATURES = ["resident_ratio", "memory_pressure", "free_memory_percent"]
MAX_DATASET_FREE_FRAMES = 219.0


def prepare_training_features(df: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame({
        "resident_ratio": df["resident_pages"] / df["working_set_size"].clip(lower=1),
        "memory_pressure": df["memory_pressure"].clip(0, 100) / 100.0,
        "free_memory_percent": df["free_frames"].clip(0, MAX_DATASET_FREE_FRAMES)
        / MAX_DATASET_FREE_FRAMES,
    }, columns=LIVE_MODEL_FEATURES)


def load_or_train_model(
    model_path: Path = MODEL_PATH,
    dataset_path: Path = DATASET_PATH,
) -> tuple[Any, str]:
    if model_path.is_file():
        return joblib.load(model_path), "pretrained"
    if not dataset_path.is_file():
        raise FileNotFoundError(
            f"No local model or training dataset found. Expected {model_path} or {dataset_path}."
        )

    dataset = pd.read_csv(dataset_path)
    required_columns = {
        "resident_pages",
        "working_set_size",
        "memory_pressure",
        "free_frames",
        "page_fault",
    }
    missing_columns = required_columns.difference(dataset.columns)
    if missing_columns:
        raise ValueError(
            f"Local training dataset is missing required columns: {', '.join(sorted(missing_columns))}."
        )

    raw_labels = dataset["page_fault"]
    if not raw_labels.isin([0, 1]).all() or raw_labels.nunique() < 2:
        raise ValueError("The local training dataset must contain both page_fault labels 0 and 1.")
    labels = raw_labels.astype(int)

    local_model = RandomForestClassifier(
        n_estimators=150,
        max_depth=10,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1,
    )
    local_model.fit(prepare_training_features(dataset), labels)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(local_model, model_path)
    return local_model, "trained_offline"


model, MODEL_STATUS = load_or_train_model()


def windows_has_admin_rights() -> bool:
    if os.name != "nt":
        return True
    try:
        import ctypes
        return bool(ctypes.windll.shell32.IsUserAnAdmin())
    except Exception:
        return False


def collect_windows_memory_counters() -> tuple[float, float]:
    if os.name != "nt":
        return 0.0, 0.0

    try:
        result = subprocess.run(
            [
                "powershell",
                "-NoProfile",
                "-NonInteractive",
                "-Command",
                "$samples = (Get-Counter -Counter '\\Memory\\Page Faults/sec','\\Memory\\Pages/sec' "
                "-ErrorAction Stop).CounterSamples; "
                "foreach ($sample in $samples) { Write-Output ($sample.Path + '=' + "
                "$sample.CookedValue.ToString([System.Globalization.CultureInfo]::InvariantCulture)) }",
            ],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return 0.0, 0.0

    if result.returncode != 0:
        return 0.0, 0.0

    counters = {"page_faults_per_sec": 0.0, "hard_faults_per_sec": 0.0}
    for line in result.stdout.splitlines():
        path, separator, value = line.rpartition("=")
        if not separator:
            continue
        normalized_path = path.lower().replace("\\", "")
        try:
            counter_value = float(value.strip())
        except ValueError:
            continue
        if normalized_path.endswith("page faults/sec"):
            counters["page_faults_per_sec"] = counter_value
        elif normalized_path.endswith("pages/sec"):
            counters["hard_faults_per_sec"] = counter_value

    return counters["page_faults_per_sec"], counters["hard_faults_per_sec"]


def collect_process_memory(limit: int | None = None) -> list[dict[str, Any]]:
    process_rows = []
    cpu_count = psutil.cpu_count(logical=True) or 1
    for process in psutil.process_iter(["pid", "name", "memory_info"]):
        try:
            memory = process.info["memory_info"]
            if memory is None or memory.vms <= 0:
                continue
            process_cpu_percent = max(0.0, float(process.cpu_percent(interval=None)))
            process_rows.append({
                "pid": int(process.info["pid"]),
                "name": process.info.get("name") or "Unknown process",
                "resident_ratio": round(max(0.0, min(1.0, memory.rss / memory.vms)), 6),
                "rss_mb": round(memory.rss / (1024 * 1024), 2),
                "cpu_percent": round(process_cpu_percent / cpu_count, 2),
            })
        except (psutil.AccessDenied, psutil.NoSuchProcess, psutil.ZombieProcess):
            continue
    process_rows.sort(key=lambda item: item["rss_mb"], reverse=True)
    return process_rows if limit is None else process_rows[:limit]


class TelemetryCollector:
    def __init__(self, sample_interval: float = 1.0, average_window: int = 1):
        self.sample_interval = sample_interval
        self.average_window = average_window
        self.lock = threading.Lock()
        self.counter_lock = threading.Lock()
        self.stop_event = threading.Event()
        self.history: deque[dict[str, Any]] = deque(maxlen=average_window)
        self.previous_processes: dict[int, str] | None = None
        self.memory_counters = {
            "page_faults_per_sec": 0.0,
            "hard_faults_per_sec": 0.0,
        }
        self.current_snapshot: dict[str, Any] = {
            "source": "laptop-monitor",
            "risk_mode": "read-only",
            "telemetry_status": "initializing",
            "page_faults_per_sec": 0.0,
            "hard_faults_per_sec": 0.0,
            "process_page_faults": {},
            "processes": [],
            "memory_percent": 0.0,
            "free_memory_percent": 0.0,
            "cpu_percent": 0.0,
            "swap_percent": 0.0,
            "disk_percent": 0.0,
            "process_count": 0,
            "privilege_status": "unknown",
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        }

    def _read_raw_snapshot(self) -> dict[str, Any]:
        try:
            vm = psutil.virtual_memory()
            swap = psutil.swap_memory()
            disk = psutil.disk_usage("/")
            cpu = psutil.cpu_percent(interval=None)
            proc_count = len(psutil.pids())
            processes = collect_process_memory()
            current_processes = {process["pid"]: process["name"] for process in processes}
            with self.lock:
                previous_processes = self.previous_processes
                self.previous_processes = current_processes
            if previous_processes is None:
                started_names = []
                stopped_names = []
            else:
                started_names = [
                    current_processes[pid] for pid in current_processes.keys() - previous_processes.keys()
                ]
                stopped_names = [
                    previous_processes[pid] for pid in previous_processes.keys() - current_processes.keys()
                ]
            with self.counter_lock:
                memory_counters = dict(self.memory_counters)

            snapshot = {
                "source": "laptop-monitor",
                "risk_mode": "read-only",
                "telemetry_status": "live",
                "cpu_percent": round(float(cpu), 2),
                "memory_percent": round(float(vm.percent), 2),
                "swap_percent": round(float(swap.percent), 2),
                "disk_percent": round(float(disk.percent), 2),
                "process_count": int(proc_count),
                "visible_process_count": len(processes),
                "visible_process_memory_mb": round(
                    sum(process["rss_mb"] for process in processes), 2
                ),
                "process_cpu_percent": round(
                    min(100.0, sum(process.get("cpu_percent", 0.0) for process in processes)), 2
                ),
                "processes_started": started_names,
                "processes_stopped": stopped_names,
                "page_faults_per_sec": round(memory_counters["page_faults_per_sec"], 2),
                "hard_faults_per_sec": round(memory_counters["hard_faults_per_sec"], 2),
                "process_page_faults": {},
                "processes": processes,
                "privilege_status": "admin" if windows_has_admin_rights() else "limited",
                "free_memory_percent": round(float(vm.available / vm.total * 100), 2),
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "rss_mb": round(float(psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)), 2),
            }
            return snapshot
        except Exception as exc:  # pragma: no cover - defensive path
            return {
                "source": "laptop-monitor",
                "risk_mode": "read-only",
                "telemetry_status": "fallback",
                "cpu_percent": 0.0,
                "memory_percent": 0.0,
                "swap_percent": 0.0,
                "disk_percent": 0.0,
                "process_count": 0,
                "visible_process_count": 0,
                "visible_process_memory_mb": 0.0,
                "process_cpu_percent": 0.0,
                "processes_started": [],
                "processes_stopped": [],
                "page_faults_per_sec": 0.0,
                "hard_faults_per_sec": 0.0,
                "process_page_faults": {},
                "processes": [],
                "privilege_status": "limited",
                "free_memory_percent": 0.0,
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "rss_mb": 0.0,
                "error": str(exc),
            }

    def _apply_moving_average(self, snapshot: dict[str, Any]) -> dict[str, Any]:
        self.history.append(snapshot)
        if len(self.history) == 1:
            return dict(snapshot)

        averaged = dict(snapshot)
        averaged["memory_percent"] = round(
            sum(item["memory_percent"] for item in self.history) / len(self.history), 2
        )
        averaged["free_memory_percent"] = round(
            sum(item["free_memory_percent"] for item in self.history) / len(self.history), 2
        )
        averaged["cpu_percent"] = round(
            sum(item["cpu_percent"] for item in self.history) / len(self.history), 2
        )
        averaged["page_faults_per_sec"] = round(
            sum(item["page_faults_per_sec"] for item in self.history) / len(self.history), 2
        )
        averaged["hard_faults_per_sec"] = round(
            sum(item["hard_faults_per_sec"] for item in self.history) / len(self.history), 2
        )
        averaged["disk_percent"] = round(
            sum(item["disk_percent"] for item in self.history) / len(self.history), 2
        )
        averaged["swap_percent"] = round(
            sum(item["swap_percent"] for item in self.history) / len(self.history), 2
        )
        averaged["telemetry_status"] = "filtered"
        return averaged

    def sample(self) -> dict[str, Any]:
        sample = self._read_raw_snapshot()
        filtered = self._apply_moving_average(sample)
        with self.lock:
            self.current_snapshot = filtered
        return filtered

    def start(self) -> None:
        if getattr(self, "_thread", None) and self._thread.is_alive():
            return

        self._thread = threading.Thread(target=self._worker, daemon=True)
        self._counter_thread = threading.Thread(target=self._counter_worker, daemon=True)
        self._thread.start()
        self._counter_thread.start()

    def stop(self) -> None:
        self.stop_event.set()

    def _worker(self) -> None:
        while not self.stop_event.is_set():
            sample_started = time.monotonic()
            self.sample()
            elapsed = time.monotonic() - sample_started
            self.stop_event.wait(max(0.0, self.sample_interval - elapsed))

    def _counter_worker(self) -> None:
        while not self.stop_event.is_set():
            page_faults, hard_faults = collect_windows_memory_counters()
            with self.counter_lock:
                self.memory_counters = {
                    "page_faults_per_sec": page_faults,
                    "hard_faults_per_sec": hard_faults,
                }
            self.stop_event.wait(2.0)

    def get_snapshot(self) -> dict[str, Any]:
        with self.lock:
            return dict(self.current_snapshot)


telemetry_collector = TelemetryCollector(sample_interval=1.0, average_window=1)


def safe_system_snapshot() -> dict[str, Any]:
    snapshot = telemetry_collector.get_snapshot()
    if snapshot.get("telemetry_status") == "initializing":
        return telemetry_collector.sample()
    return snapshot


def build_live_model_input(snapshot: dict[str, Any]) -> pd.DataFrame:
    memory_pressure = max(0.0, min(100.0, float(snapshot.get("memory_percent", 0.0)))) / 100.0
    free_memory_percent = max(0.0, min(100.0, float(snapshot.get("free_memory_percent", 0.0)))) / 100.0
    rows = []
    for process in snapshot.get("processes", []):
        resident_ratio = max(0.0, min(0.999, float(process.get("resident_ratio", 0.0))))
        rows.append({
            "resident_ratio": resident_ratio,
            "memory_pressure": memory_pressure,
            "free_memory_percent": free_memory_percent,
        })
    return pd.DataFrame(rows, columns=LIVE_MODEL_FEATURES)


def score_live_processes(snapshot: dict[str, Any]) -> list[dict[str, Any]]:
    if model is None:
        return []
    features = build_live_model_input(snapshot)
    processes = snapshot.get("processes", [])
    if features.empty:
        return []

    probabilities = model.predict_proba(features)[:, 1]
    scored = []
    for process, probability in zip(processes, probabilities):
        risk_score = round(float(probability) * 100, 2)
        scored.append({
            "process_name": process["name"],
            "pid": process["pid"],
            "rss_mb": process["rss_mb"],
            "cpu_percent": process.get("cpu_percent", 0.0),
            "risk_score": risk_score,
            "risk_level": classify_risk_level(risk_score),
        })
    scored.sort(key=lambda item: item["risk_score"], reverse=True)
    return scored


def classify_risk_level(score: float) -> str:
    if score >= 75:
        return "high"
    if score >= 40:
        return "medium"
    return "low"


def get_refill_policy(risk_score: float) -> dict[str, Any]:
    if risk_score >= 90:
        return {"mode": "paused", "tokens_per_cycle": 0, "message": "Critical predicted risk: refill paused in simulation."}
    if risk_score >= 75:
        return {"mode": "slow", "tokens_per_cycle": 1, "message": "High predicted risk: refill slowed in simulation."}
    if risk_score >= 40:
        return {"mode": "moderate", "tokens_per_cycle": 2, "message": "Moderate predicted risk: partial refill in simulation."}
    return {"mode": "steady", "tokens_per_cycle": 3, "message": "Low predicted risk: normal refill in simulation."}


def build_token_booth(process_predictions: list[dict[str, Any]]) -> dict[str, Any]:
    process_rows: list[dict[str, Any]] = []

    priority_order = sorted(
        process_predictions,
        key=lambda process: (process["risk_score"], process["process_name"].lower(), process["pid"]),
    )
    for priority_rank, process in enumerate(priority_order, start=1):
        process_risk = float(process["risk_score"])
        refill_policy = get_refill_policy(process_risk)
        process_rows.append({
            "priority_rank": priority_rank,
            "process_name": process["process_name"],
            "pid": process["pid"],
            "memory_mb": process["rss_mb"],
            "risk_score": round(process_risk, 2),
            "risk_level": process["risk_level"],
            "tokens_remaining": max(0, 10 - int(process_risk / 10)),
            "tokens_total": 10,
            "refill_rate": refill_policy["tokens_per_cycle"],
            "refill_mode": refill_policy["mode"],
            "refill_message": refill_policy["message"],
            "refill_paused": refill_policy["tokens_per_cycle"] == 0,
        })

    highest_risk = max((item["risk_score"] for item in process_rows), default=0.0)
    risk_level = classify_risk_level(highest_risk)
    refill_policy = get_refill_policy(highest_risk)
    return {
        "tokens_per_process": 10,
        "risk_classification": risk_level,
        "refill_rate": refill_policy["tokens_per_cycle"],
        "refill_mode": refill_policy["mode"],
        "processes": process_rows,
        "scheduler_order": [item["process_name"] for item in process_rows],
        "priority_rule": "lowest_predicted_risk_first",
        "mode": "read-only-simulation",
        "controls_system_processes": False,
    }


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["GET"])
def predict():
    if model is None:
        return jsonify({
            "error": "Model not found. Run train_model.py first."
        }), 500

    snapshot = safe_system_snapshot()
    process_predictions = score_live_processes(snapshot)
    risk_score = max((item["risk_score"] for item in process_predictions), default=None)
    category = classify_risk_level(risk_score) if risk_score is not None else "unavailable"
    token_booth = build_token_booth(process_predictions)

    return jsonify({
        "risk_score": risk_score,
        "category": category,
        "score_type": "live_model_estimate",
        "execution_mode": "offline",
        "model_status": MODEL_STATUS,
        "score_note": "Live process memory ratios scored locally by a model trained on synthetic page_fault labels; not a calibrated real-world probability.",
        "telemetry_status": snapshot.get("telemetry_status", "unavailable"),
        "snapshot": snapshot,
        "process_predictions": process_predictions,
        "token_booth": token_booth,
    })


telemetry_collector.start()

if __name__ == "__main__":
    app.run(debug=True, host="127.0.0.1", port=5000) 
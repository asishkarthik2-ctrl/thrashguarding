import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
import joblib

MODEL_PATH = "page_fault_model.pkl"
DATASET_PATH = Path("page_fault_dataset.csv")
METRICS_PATH = Path("page_fault_metrics.json")
COMPARISON_PATH = Path("model_comparison.json")
LIVE_FEATURE_COLUMNS = ["resident_ratio", "memory_pressure", "free_memory_percent"]
MAX_DATASET_FREE_FRAMES = 219.0


def generate_dataset(n_samples=20000, random_state=42):
    rng = np.random.default_rng(random_state)
    rows = []

    for _ in range(n_samples):
        process_id = int(rng.integers(1, 25))
        page_number = int(rng.integers(1, 5000))
        access_type = rng.choice(["read", "write"])
        working_set_size = int(rng.integers(25, 700))
        resident_pages = int(rng.integers(10, max(20, working_set_size)))
        free_frames = int(rng.integers(2, 220))
        memory_pressure = int(rng.integers(10, 100))
        page_age = int(rng.integers(0, 800))
        locality = int(rng.integers(0, 100))
        recent_accesses = int(rng.integers(0, 80))

        resident_ratio = resident_pages / max(working_set_size, 1)
        not_resident = 1 if resident_ratio < 0.55 else 0
        write_penalty = 1 if access_type == "write" and (memory_pressure > 60 or free_frames < 20) else 0
        age_penalty = 1 if page_age > 350 and memory_pressure > 55 else 0
        locality_penalty = 1 if locality < 35 and recent_accesses < 10 else 0
        pressure_penalty = max(0, (memory_pressure - 60) / 40)
        frame_penalty = max(0, (25 - free_frames) / 25)

        fault_score = (
            0.38 * not_resident
            + 0.18 * write_penalty
            + 0.14 * age_penalty
            + 0.12 * locality_penalty
            + 0.10 * pressure_penalty
            + 0.08 * frame_penalty
        )

        page_fault = 1 if fault_score > 0.5 else 0
        access_code = 1 if access_type == "write" else 0

        rows.append([
            process_id,
            page_number,
            access_code,
            working_set_size,
            resident_pages,
            free_frames,
            memory_pressure,
            page_age,
            locality,
            recent_accesses,
            page_fault,
        ])

    df = pd.DataFrame(rows, columns=[
        "process_id",
        "page_number",
        "access_type",
        "working_set_size",
        "resident_pages",
        "free_frames",
        "memory_pressure",
        "page_age",
        "locality",
        "recent_accesses",
        "page_fault",
    ])
    return df


def ensure_dataset():
    if DATASET_PATH.exists():
        return pd.read_csv(DATASET_PATH)

    df = generate_dataset()
    df.to_csv(DATASET_PATH, index=False)
    print(f"Generated dataset saved to {DATASET_PATH}")
    return df


def prepare_live_features(df):
    return pd.DataFrame({
        "resident_ratio": df["resident_pages"] / df["working_set_size"].clip(lower=1),
        "memory_pressure": df["memory_pressure"].clip(0, 100) / 100.0,
        "free_memory_percent": df["free_frames"].clip(0, MAX_DATASET_FREE_FRAMES) / MAX_DATASET_FREE_FRAMES,
    })


def evaluate_model(model, X_train, X_test, y_train, y_test, name):
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(y_test, predictions, zero_division=0)
    recall = recall_score(y_test, predictions, zero_division=0)
    f1 = f1_score(y_test, predictions, zero_division=0)
    cm = confusion_matrix(y_test, predictions).tolist()

    metrics = {
        "model": name,
        "accuracy": round(float(accuracy), 4),
        "precision": round(float(precision), 4),
        "recall": round(float(recall), 4),
        "f1_score": round(float(f1), 4),
        "confusion_matrix": cm,
    }

    print(f"\n{name} accuracy: {round(accuracy * 100, 2)}%")
    print(classification_report(y_test, predictions, target_names=["No Fault", "Page Fault"]))

    return metrics


def compare_models(df):
    X = prepare_live_features(df)
    y = df["page_fault"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    models = {
        "RandomForest": RandomForestClassifier(
            n_estimators=300,
            max_depth=10,
            min_samples_leaf=2,
            random_state=42,
        ),
        "XGBoost": XGBClassifier(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            objective="binary:logistic",
            eval_metric="logloss",
            random_state=42,
        ),
        "LogisticRegression": LogisticRegression(max_iter=2000, random_state=42),
    }

    results = []
    best_model = None
    best_name = None
    best_f1 = -1.0

    for name, model in models.items():
        metrics = evaluate_model(model, X_train, X_test, y_train, y_test, name)
        results.append(metrics)

        if metrics["f1_score"] > best_f1:
            best_f1 = metrics["f1_score"]
            best_name = name
            best_model = model
            best_metrics = metrics

    with COMPARISON_PATH.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    best_metrics["features"] = LIVE_FEATURE_COLUMNS
    best_metrics["training_labels"] = "synthetic page_fault_dataset.csv labels"
    with METRICS_PATH.open("w", encoding="utf-8") as f:
        json.dump(best_metrics, f, indent=2)

    print(f"\nBest model: {best_name} with F1 {round(best_f1 * 100, 2)}%")
    joblib.dump(best_model, MODEL_PATH)
    print(f"Saved best model as {MODEL_PATH}")
    print(f"Saved comparison report as {COMPARISON_PATH}")

    return best_model


def main():
    df = ensure_dataset()
    compare_models(df)


if __name__ == "__main__":
    main()

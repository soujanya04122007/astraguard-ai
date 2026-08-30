"""
src/anomaly_detector.py
────────────────────────────────────────────────────────────────────────────
AstraGuard AI — Anomaly Detection Module
IBM AI Builders Challenge, August 2026: Advance Space Exploration with AI

This module provides a clean, importable API for loading telemetry data,
training an IsolationForest anomaly detector, scoring new readings, and
persisting both the trained model and the annotated results.

It is intentionally side-effect-free when imported so that a Streamlit
dashboard can call individual functions without triggering training.

NOTE: All telemetry is SIMULATED data created for an AI prototype.
      It is NOT real NASA or agency telemetry.
"""

import os
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# ── Constants ─────────────────────────────────────────────────────────────────

# Telemetry columns used as model features (timestamp and derived columns excluded)
FEATURE_COLS = [
    "temperature",
    "battery_voltage",
    "power_consumption",
    "radiation_level",
    "signal_strength",
    "fuel_level",
    "solar_output",
]

# IsolationForest hyper-parameters
# contamination matches our known ~5 % anomaly injection rate
IF_PARAMS = dict(
    n_estimators=200,       # more trees → more stable anomaly scores
    contamination=0.05,     # expected fraction of outliers in the dataset
    max_features=1.0,       # use all features per tree
    random_state=42,
    n_jobs=-1,              # parallelise across all available CPU cores
)

# Default file paths (callers may override)
DEFAULT_DATA_PATH   = os.path.join("data",   "telemetry.csv")
DEFAULT_MODEL_PATH  = os.path.join("models", "anomaly_model.pkl")
DEFAULT_RESULT_PATH = os.path.join("data",   "anomaly_results.csv")


# ── Data loading ──────────────────────────────────────────────────────────────

def load_telemetry(path: str = DEFAULT_DATA_PATH) -> pd.DataFrame:
    """
    Read the telemetry CSV and parse the timestamp column.

    Returns a DataFrame with a proper DatetimeIndex-friendly 'timestamp'
    column so callers can filter or resample by time if needed.
    """
    df = pd.read_csv(path, parse_dates=["timestamp"])
    return df


# ── Model building ────────────────────────────────────────────────────────────

def build_pipeline() -> Pipeline:
    """
    Construct a sklearn Pipeline:
        1. StandardScaler  — zero-mean, unit-variance normalisation.
           IsolationForest is distance-based internally, so scaling ensures
           no single high-magnitude channel (e.g. power_consumption in Watts)
           dominates the anomaly score.
        2. IsolationForest — unsupervised outlier detection via random
           partitioning trees.
    """
    return Pipeline([
        ("scaler", StandardScaler()),
        ("iforest", IsolationForest(**IF_PARAMS)),
    ])


# ── Training ──────────────────────────────────────────────────────────────────

def train(df: pd.DataFrame) -> tuple[Pipeline, pd.DataFrame]:
    """
    Fit the anomaly detection pipeline on the telemetry DataFrame.

    Returns
    -------
    pipeline : fitted sklearn Pipeline
    results  : original DataFrame augmented with three new columns:
                 anomaly_score  — raw IsolationForest decision score
                                  (more negative = more anomalous)
                 anomaly_flag   — sklearn convention: -1 anomaly, +1 normal
                 label          — human-readable "Anomaly" / "Normal"
    """
    X = df[FEATURE_COLS].copy()

    # Fit the full pipeline (scale → forest) on all 1,000 records
    pipeline = build_pipeline()
    pipeline.fit(X)

    # decision_function returns the mean depth-based anomaly score.
    # Lower (more negative) scores indicate stronger anomalies.
    scores = pipeline.decision_function(X)

    # predict() returns -1 for anomalies and +1 for normal observations
    flags = pipeline.predict(X)

    results = df.copy()
    results["anomaly_score"] = np.round(scores, 6)
    results["anomaly_flag"]  = flags
    results["label"]         = np.where(flags == -1, "Anomaly", "Normal")

    return pipeline, results


# ── Persistence ───────────────────────────────────────────────────────────────

def save_model(pipeline: Pipeline, path: str = DEFAULT_MODEL_PATH) -> None:
    """Serialise the fitted pipeline to disk using joblib."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(pipeline, path)


def load_model(path: str = DEFAULT_MODEL_PATH) -> Pipeline:
    """Deserialise and return a previously saved pipeline."""
    return joblib.load(path)


def save_results(results: pd.DataFrame, path: str = DEFAULT_RESULT_PATH) -> None:
    """Write the annotated telemetry DataFrame to CSV."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    results.to_csv(path, index=False)


# ── Inference helper (for dashboard / streaming use) ─────────────────────────

def score_dataframe(pipeline: Pipeline, df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply a pre-fitted pipeline to any telemetry DataFrame and return it
    with anomaly_score, anomaly_flag, and label columns appended.

    Useful for scoring new batches of telemetry without retraining.
    """
    X = df[FEATURE_COLS].copy()
    scores = pipeline.decision_function(X)
    flags  = pipeline.predict(X)

    out = df.copy()
    out["anomaly_score"] = np.round(scores, 6)
    out["anomaly_flag"]  = flags
    out["label"]         = np.where(flags == -1, "Anomaly", "Normal")
    return out


# ── Summary helper ────────────────────────────────────────────────────────────

def detection_summary(results: pd.DataFrame) -> dict:
    """
    Return a plain dictionary of key detection statistics.
    Convenient for dashboard KPI cards.
    """
    total     = len(results)
    n_anomaly = int((results["label"] == "Anomaly").sum())
    n_normal  = total - n_anomaly

    return {
        "total_records":   total,
        "normal_count":    n_normal,
        "anomaly_count":   n_anomaly,
        "anomaly_rate_pct": round(n_anomaly / total * 100, 2),
        "min_score":       round(float(results["anomaly_score"].min()), 4),
        "max_score":       round(float(results["anomaly_score"].max()), 4),
    }

"""
train_anomaly_model.py
──────────────────────
AstraGuard AI — Model Training Entry Point
IBM AI Builders Challenge, August 2026: Advance Space Exploration with AI

Run this script to:
  1. Load the simulated telemetry from data/telemetry.csv
  2. Train an IsolationForest anomaly detection pipeline
  3. Save the fitted model   → models/anomaly_model.pkl
  4. Save annotated results  → data/anomaly_results.csv

Usage:
    python train_anomaly_model.py

NOTE: All telemetry is SIMULATED data for an AI prototype.
      It is NOT real NASA or agency telemetry.
"""

import sys
import os

# Allow imports from src/ regardless of working directory
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from anomaly_detector import (
    load_telemetry,
    train,
    save_model,
    save_results,
    detection_summary,
    DEFAULT_DATA_PATH,
    DEFAULT_MODEL_PATH,
    DEFAULT_RESULT_PATH,
    FEATURE_COLS,
)

# ── 1. Load telemetry ─────────────────────────────────────────────────────────
print("AstraGuard AI — Anomaly Detection Training")
print("=" * 50)
print(f"  Loading telemetry : {DEFAULT_DATA_PATH}")

df = load_telemetry(DEFAULT_DATA_PATH)
print(f"  Records loaded    : {len(df):,}")
print(f"  Features used     : {FEATURE_COLS}")

# ── 2. Train ──────────────────────────────────────────────────────────────────
print("\n  Training IsolationForest pipeline ...")
pipeline, results = train(df)
print("  Training complete.")

# ── 3. Save model ─────────────────────────────────────────────────────────────
save_model(pipeline, DEFAULT_MODEL_PATH)
print(f"\n  Model saved       : {DEFAULT_MODEL_PATH}")

# ── 4. Save annotated results ─────────────────────────────────────────────────
save_results(results, DEFAULT_RESULT_PATH)
print(f"  Results saved     : {DEFAULT_RESULT_PATH}")

# ── 5. Console summary ────────────────────────────────────────────────────────
summary = detection_summary(results)

print("\n--- Detection Summary ---")
print(f"  Total records     : {summary['total_records']:,}")
print(f"  Normal            : {summary['normal_count']:,}")
print(f"  Anomalies flagged : {summary['anomaly_count']:,}  ({summary['anomaly_rate_pct']}%)")
print(f"  Score range       : {summary['min_score']}  to  {summary['max_score']}")

# Show a sample of the detected anomalies with their scores
anomalies = results[results["label"] == "Anomaly"].sort_values("anomaly_score")
print(f"\n--- Top 10 Most Anomalous Records ---")
display_cols = ["timestamp", "temperature", "battery_voltage",
                "power_consumption", "radiation_level",
                "signal_strength", "solar_output", "anomaly_score", "label"]
print(anomalies[display_cols].head(10).to_string(index=False))

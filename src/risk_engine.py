"""
src/risk_engine.py
──────────────────────────────────────────────────────────────────────────────
AstraGuard AI — Mission Risk Index Engine
IBM AI Builders Challenge, August 2026: Advance Space Exploration with AI

PROTOTYPE DISCLAIMER
────────────────────
This module computes a Mission Risk Index (MRI) purely as a decision-support
prototype for an AI demonstration. It is NOT a certified aerospace risk
standard, NOT validated against real mission data, and must NOT be used for
actual spacecraft operations or safety-critical decisions.

PURPOSE
───────
Translate raw telemetry values and IsolationForest anomaly scores into a
transparent, explainable 0–100 integer risk index with an associated severity
level. Each risk contribution is individually traceable to a specific telemetry
channel so that operators can see exactly why the risk is elevated.

SCORING ARCHITECTURE
────────────────────
Risk is built from two orthogonal signals per channel:

  1. THRESHOLD EXCEEDANCE — rule-based: how far is the value outside its
     known safe operating envelope?  Produces a 0–100 component score.

  2. ANOMALY WEIGHT BOOST — model-based: if the IsolationForest also flagged
     the row as an anomaly, the threshold component gets a multiplied boost,
     ensuring ML detections that don't break a fixed threshold still contribute.

Final MRI = weighted sum of per-channel scores, clipped to [0, 100].

Channel weights reflect operational criticality (power and battery failures
are mission-ending; solar and comms are serious but recoverable):

  battery_voltage    0.25   highest — undervoltage kills all subsystems
  temperature        0.20   high    — thermal runaway risks hardware
  power_consumption  0.18   high    — abnormal draw signals fault cascade
  radiation_level    0.15   medium  — crew/instrument exposure
  signal_strength    0.12   medium  — loss of comms is serious
  solar_output       0.10   lower   — short eclipses are manageable

Risk levels (MRI thresholds):
  0  – 30  → LOW       (nominal operations)
  31 – 60  → MEDIUM    (operator attention warranted)
  61 – 80  → HIGH      (immediate investigation required)
  81 – 100 → CRITICAL  (mission-threatening condition)

USAGE
─────
  from src.risk_engine import assess_row, assess_dataframe, batch_summary

  # Single row (a plain dict or a 1-row DataFrame):
  result = assess_row(row, anomaly_flag=-1, anomaly_score=-0.13)

  # Whole DataFrame (already annotated by anomaly_detector):
  risk_df = assess_dataframe(annotated_df)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Any

# ─────────────────────────────────────────────────────────────────────────────
# 1.  Safe-operating envelopes  (min_normal, max_normal)
#
#     Values inside [min_normal, max_normal] score 0 on the threshold axis.
#     Values outside begin accumulating risk linearly until the channel's
#     "max exceedance" boundary, where the score reaches 100.
#
#     Boundaries are derived directly from the generate_telemetry.py baselines
#     plus a small tolerance so minor natural variation doesn't raise alerts.
# ─────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class ChannelEnvelope:
    """Defines the safe and danger boundaries for one telemetry channel."""
    name:          str
    min_normal:    float          # lower bound of safe operating range
    max_normal:    float          # upper bound of safe operating range
    min_danger:    float          # value at which channel score reaches 100
    max_danger:    float          # value at which channel score reaches 100
    weight:        float          # contribution weight in final MRI (sum = 1.0)
    direction:     str            # "high" | "low" | "both"
    unit:          str            # display unit string
    description:   str            # human-readable fault description

    def threshold_score(self, value: float) -> float:
        """
        Map `value` to a 0–100 exceedance score on this channel.

        The mapping is piecewise linear:
          inside  [min_normal, max_normal]          → 0
          outside, towards danger boundary           → linear 0 → 100
          at or beyond danger boundary               → 100

        Direction controls which side(s) are penalised.
        """
        score = 0.0

        # High-side exceedance
        if self.direction in ("high", "both") and value > self.max_normal:
            span = self.max_danger - self.max_normal
            score = max(score, min(100.0, (value - self.max_normal) / span * 100.0))

        # Low-side exceedance
        if self.direction in ("low", "both") and value < self.min_normal:
            span = self.min_normal - self.min_danger
            score = max(score, min(100.0, (self.min_normal - value) / span * 100.0))

        return score


# ─────────────────────────────────────────────────────────────────────────────
# 2.  Channel definitions
#     Each envelope is calibrated against the generate_telemetry.py v2 normal
#     ranges:  temperature 1–27 °C, battery 27.4–29.1 V, power 65–135 W,
#              radiation 0.28–0.86 mSv/h, signal −80 to −63 dBm,
#              solar 63–114 W
# ─────────────────────────────────────────────────────────────────────────────

ENVELOPES: dict[str, ChannelEnvelope] = {
    "temperature": ChannelEnvelope(
        name="temperature",
        min_normal=-5.0,   max_normal=30.0,    # normal instrument-bay range
        min_danger=-25.0,  max_danger=85.0,    # heater failure / thermal runaway
        weight=0.20,
        direction="both",
        unit="degC",
        description="Thermal anomaly",
    ),
    "battery_voltage": ChannelEnvelope(
        name="battery_voltage",
        min_normal=27.0,   max_normal=29.5,    # nominal 28 V bus
        min_danger=18.0,   max_danger=31.0,    # cell failure / overcharge
        weight=0.25,
        direction="both",
        unit="V",
        description="Battery voltage anomaly",
    ),
    "power_consumption": ChannelEnvelope(
        name="power_consumption",
        min_normal=55.0,   max_normal=145.0,   # nominal subsystem draw
        min_danger=30.0,   max_danger=270.0,   # fault cascade / short circuit
        weight=0.18,
        direction="both",
        unit="W",
        description="Power consumption anomaly",
    ),
    "radiation_level": ChannelEnvelope(
        name="radiation_level",
        min_normal=0.10,   max_normal=1.30,    # interplanetary cruise baseline
        min_danger=0.05,   max_danger=8.0,     # solar particle event
        weight=0.15,
        direction="high",
        unit="mSv/h",
        description="Radiation anomaly",
    ),
    "signal_strength": ChannelEnvelope(
        name="signal_strength",
        min_normal=-82.0,  max_normal=-55.0,   # good link margin
        min_danger=-125.0, max_danger=-50.0,   # deep fade / link loss
        weight=0.12,
        direction="low",                        # lower dBm = worse signal
        unit="dBm",
        description="Communication signal anomaly",
    ),
    "solar_output": ChannelEnvelope(
        name="solar_output",
        min_normal=55.0,   max_normal=125.0,   # solar array in sunlight
        min_danger=5.0,    max_danger=150.0,   # complete occlusion / degradation
        weight=0.10,
        direction="low",
        unit="W",
        description="Solar output anomaly",
    ),
}

# Sanity check: weights should sum to 1.0
assert abs(sum(e.weight for e in ENVELOPES.values()) - 1.0) < 1e-9, \
    "Channel weights must sum to 1.0"

# Channels actively used in MRI (fuel_level excluded — it depletes normally)
RISK_CHANNELS = list(ENVELOPES.keys())

# Boost multiplier applied to a channel score when the ML model also flags
# the row as an anomaly.  Keeps rule scores dominant but rewards ML agreement.
ANOMALY_BOOST = 1.35

# ─────────────────────────────────────────────────────────────────────────────
# MRI SCALING
# ─────────────────────────────────────────────────────────────────────────────
# Each channel produces a 0–100 threshold_score.  Rather than summing weighted
# fractions (which caps the total at 100 × max_weight ≈ 25), we take the
# WEIGHTED MAXIMUM approach:
#
#   MRI = max( channel_i_score × weight_i × SCALE ) clamped to [0, 100]
#
# But for a more nuanced composite that lets multiple moderate faults combine:
#
#   MRI = sum( score_i × weight_i × SCALE ) clamped to [0, 100]
#
# SCALE maps "one channel at 100 %, its full weight" to the target MRI.
# We want one fully-saturated critical channel (battery, weight=0.25) to reach
# CRITICAL (≥81).  So:  100 × 0.25 × SCALE ≥ 81  →  SCALE ≥ 3.24.
# Using SCALE = 3.5 lets a fully-faulted battery hit 87 (CRITICAL) while a
# fully-faulted solar panel (weight=0.10) reaches 35 (MEDIUM), which is
# reasonable — solar loss is serious but not immediately mission-ending.
SCALE = 3.5


# ─────────────────────────────────────────────────────────────────────────────
# 3.  Risk level thresholds
# ─────────────────────────────────────────────────────────────────────────────

RISK_LEVELS = [
    (81, "CRITICAL"),
    (61, "HIGH"),
    (31, "MEDIUM"),
    ( 0, "LOW"),
]

def _risk_level(score: float) -> str:
    """Map a 0–100 MRI score to its named severity level."""
    for threshold, label in RISK_LEVELS:
        if score >= threshold:
            return label
    return "LOW"


# ─────────────────────────────────────────────────────────────────────────────
# 4.  Core assessment logic
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ChannelContribution:
    """Holds the scored breakdown for one telemetry channel."""
    channel:         str
    value:           float
    unit:            str
    threshold_score: float    # 0–100 rule-based exceedance
    weighted_score:  float    # after weight and optional anomaly boost
    is_contributing: bool     # True when this channel adds meaningful risk
    description:     str      # fault description string


@dataclass
class RiskAssessment:
    """
    Full risk assessment result for one telemetry reading.

    Attributes
    ----------
    risk_score         : int   — Mission Risk Index [0, 100]
    risk_level         : str   — "LOW" | "MEDIUM" | "HIGH" | "CRITICAL"
    anomaly_detected   : bool  — whether the ML model flagged this row
    anomaly_score      : float — raw IsolationForest decision score
    contributions      : list  — per-channel ChannelContribution objects,
                                 sorted by weighted_score descending
    top_factors        : list  — human-readable list of main risk drivers
    explanation        : str   — one-paragraph narrative explanation
    """
    risk_score:       int
    risk_level:       str
    anomaly_detected: bool
    anomaly_score:    float
    contributions:    list[ChannelContribution] = field(default_factory=list)
    top_factors:      list[str]                 = field(default_factory=list)
    explanation:      str                       = ""


def assess_row(
    row: dict[str, Any] | pd.Series,
    anomaly_flag:  int   = 1,
    anomaly_score: float = 0.0,
) -> RiskAssessment:
    """
    Compute the Mission Risk Index for a single telemetry reading.

    Parameters
    ----------
    row           : dict or pd.Series with the 7 telemetry channel values.
    anomaly_flag  : sklearn IsolationForest output (-1 = anomaly, +1 = normal).
    anomaly_score : decision_function score (more negative = more anomalous).

    Returns
    -------
    RiskAssessment dataclass with score, level, contributions, and explanation.
    """
    is_anomaly = (anomaly_flag == -1)
    contributions: list[ChannelContribution] = []
    total_weighted = 0.0

    for ch_name, env in ENVELOPES.items():
        value = float(row[ch_name])
        t_score = env.threshold_score(value)

        # Apply anomaly boost: if ML also flagged this row, scale up all
        # non-zero channel scores so ML-detected faults get heavier weight.
        if is_anomaly and t_score > 0:
            effective_score = min(100.0, t_score * ANOMALY_BOOST)
        else:
            effective_score = t_score

        # Weighted + scaled contribution to the final MRI.
        # SCALE converts fractional weights back to the 0–100 MRI space so that
        # a fully-saturated critical channel reaches CRITICAL (≥81).
        w_score = effective_score * env.weight * SCALE
        total_weighted += w_score

        contributions.append(ChannelContribution(
            channel=ch_name,
            value=round(value, 3),
            unit=env.unit,
            threshold_score=round(t_score, 2),
            weighted_score=round(w_score, 4),
            is_contributing=(t_score > 0),
            description=env.description,
        ))

    # Anomaly-score penalty: when ML flags an anomaly but no single channel
    # broke its threshold (e.g. subtle multi-variate fault), add a small
    # baseline risk so the anomaly is never silently ignored.
    ml_only_bump = 0.0
    if is_anomaly and total_weighted < 5.0:
        # Normalise the IF score to a 0–15 bump (score range is ~[-0.15, 0.15])
        ml_only_bump = min(15.0, max(0.0, abs(anomaly_score) / 0.15 * 15.0))
    total_weighted += ml_only_bump

    # Clip final score to [0, 100] and convert to integer
    mri = int(round(min(100.0, max(0.0, total_weighted))))

    # Sort contributing channels by their weighted score descending
    contributions.sort(key=lambda c: c.weighted_score, reverse=True)

    # Top factors: channels with a non-zero threshold score
    top_factors = [
        f"{c.description} ({c.channel}: {c.value} {c.unit})"
        for c in contributions if c.is_contributing
    ]

    # Build a short narrative explanation
    level = _risk_level(mri)
    if not top_factors:
        if is_anomaly:
            explanation = (
                f"MRI {mri} ({level}). IsolationForest flagged this reading as "
                f"anomalous (score {anomaly_score:.4f}) but no single channel "
                "broke its individual threshold. This may indicate a subtle "
                "multi-variate deviation warranting closer inspection."
            )
        else:
            explanation = f"MRI {mri} ({level}). All channels within normal operating envelopes."
    else:
        factor_str = "; ".join(top_factors[:3])
        explanation = (
            f"MRI {mri} ({level}). Primary risk drivers: {factor_str}. "
            f"{'IsolationForest independently confirmed this anomaly. ' if is_anomaly else ''}"
            "PROTOTYPE — not for operational use."
        )

    return RiskAssessment(
        risk_score=mri,
        risk_level=level,
        anomaly_detected=is_anomaly,
        anomaly_score=round(anomaly_score, 6),
        contributions=contributions,
        top_factors=top_factors,
        explanation=explanation,
    )


# ─────────────────────────────────────────────────────────────────────────────
# 5.  Batch processing — operates on the annotated DataFrame produced by
#     anomaly_detector.score_dataframe() or anomaly_detector.train()
# ─────────────────────────────────────────────────────────────────────────────

def assess_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply assess_row to every row of an annotated telemetry DataFrame.

    The DataFrame must contain:
      - the 7 telemetry channel columns (see RISK_CHANNELS)
      - 'anomaly_flag'  (from IsolationForest predict)
      - 'anomaly_score' (from IsolationForest decision_function)

    Returns the original DataFrame with these columns appended:
      risk_score      int     — MRI value [0, 100]
      risk_level      str     — severity label
      top_factors     str     — pipe-separated list of contributing factors
    """
    risk_scores  = []
    risk_levels  = []
    top_factors_list = []

    for _, row in df.iterrows():
        assessment = assess_row(
            row,
            anomaly_flag=int(row.get("anomaly_flag", 1)),
            anomaly_score=float(row.get("anomaly_score", 0.0)),
        )
        risk_scores.append(assessment.risk_score)
        risk_levels.append(assessment.risk_level)
        top_factors_list.append(" | ".join(assessment.top_factors) if assessment.top_factors else "None")

    out = df.copy()
    out["risk_score"]   = risk_scores
    out["risk_level"]   = risk_levels
    out["top_factors"]  = top_factors_list
    return out


# ─────────────────────────────────────────────────────────────────────────────
# 6.  Batch summary — convenient for dashboard KPI cards
# ─────────────────────────────────────────────────────────────────────────────

def batch_summary(risk_df: pd.DataFrame) -> dict:
    """
    Aggregate risk statistics across all rows in a risk-assessed DataFrame.

    Returns a dict suitable for populating dashboard KPI cards:
      current_risk_score   — most recent record's MRI
      current_risk_level   — most recent record's level
      max_risk_score       — peak MRI in the dataset
      mean_risk_score      — average MRI (normal baseline indicator)
      critical_count       — rows at CRITICAL level
      high_count           — rows at HIGH level
      medium_count         — rows at MEDIUM level
      low_count            — rows at LOW level
      top_channel          — channel that contributed most risk overall
    """
    level_counts = risk_df["risk_level"].value_counts().to_dict()

    # Identify the channel that appeared most in top_factors
    all_factors = " | ".join(risk_df["top_factors"].dropna())
    channel_freq: dict[str, int] = {}
    for ch in RISK_CHANNELS:
        channel_freq[ch] = all_factors.count(ch)
    top_channel = max(channel_freq, key=channel_freq.get) if any(channel_freq.values()) else "N/A"

    return {
        "current_risk_score": int(risk_df["risk_score"].iloc[-1]),
        "current_risk_level": risk_df["risk_level"].iloc[-1],
        "max_risk_score":     int(risk_df["risk_score"].max()),
        "mean_risk_score":    round(float(risk_df["risk_score"].mean()), 1),
        "critical_count":     int(level_counts.get("CRITICAL", 0)),
        "high_count":         int(level_counts.get("HIGH",     0)),
        "medium_count":       int(level_counts.get("MEDIUM",   0)),
        "low_count":          int(level_counts.get("LOW",      0)),
        "top_channel":        top_channel,
    }


# ─────────────────────────────────────────────────────────────────────────────
# 7.  Convenience: level → display colour (for Streamlit / Plotly callers)
# ─────────────────────────────────────────────────────────────────────────────

LEVEL_COLORS: dict[str, str] = {
    "LOW":      "#22c55e",   # green
    "MEDIUM":   "#f59e0b",   # amber
    "HIGH":     "#ef4444",   # red
    "CRITICAL": "#7c3aed",   # purple
}

def level_color(level: str) -> str:
    """Return a CSS hex colour string for the given risk level label."""
    return LEVEL_COLORS.get(level, "#6b7280")

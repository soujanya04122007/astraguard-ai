"""
generate_telemetry.py
─────────────────────────────────────────────────────────────────────────────
AstraGuard AI — Simulated Spacecraft Telemetry Generator
IBM AI Builders Challenge, August 2026: Advance Space Exploration with AI

Generates 1,000 timestamped records (5-minute cadence) representing a
fictional spacecraft mission. The data is structured in two layers:

  1. NORMAL BASELINE — each channel follows a smooth, physically plausible
     signal built from a slowly-drifting trend + low-amplitude sinusoidal
     variation + small Gaussian sensor noise.

  2. ANOMALY EVENTS — approximately 5 % of records (≈50 rows) are corrupted
     by 6 distinct fault types, each injected as a SHORT CONTIGUOUS WINDOW
     (3-8 readings) rather than isolated random spikes. This mimics how real
     faults develop and persist for a brief period before being resolved.

     Anomaly types:
       thermal      — temperature spike (heater runaway / solar exposure)
       power        — power-consumption surge (subsystem fault)
       battery      — voltage drop (cell failure / discharge event)
       radiation    — radiation burst (solar particle event / belt crossing)
       comms        — signal-strength degradation (antenna / link loss)
       solar        — solar-output drop (panel occlusion / degradation)

IMPORTANT: This is SIMULATED telemetry created for an AI prototype.
           It is NOT real NASA, ESA, or any space-agency telemetry data.

Output: data/telemetry.csv  (exactly 1,000 rows, 8 columns, no label column)
"""

import os
import numpy as np
import pandas as pd

# ─────────────────────────────────────────────────────────────────────────────
# 0. Reproducibility & constants
# ─────────────────────────────────────────────────────────────────────────────

SEED       = 42
rng        = np.random.default_rng(SEED)

N          = 1_000          # total records
CADENCE    = "5min"         # one reading every 5 minutes
START_TS   = "2026-01-01 00:00:00"
OUT_PATH   = os.path.join("data", "telemetry.csv")

t = np.arange(N, dtype=float)   # integer time index used for wave shaping


# ─────────────────────────────────────────────────────────────────────────────
# 1. Helper — smooth baseline signal
#    Combines a slow sinusoidal drift with gentle Gaussian sensor noise.
#    All channels look like they come from real sensors, not white noise.
# ─────────────────────────────────────────────────────────────────────────────

def smooth_signal(
    midpoint: float,
    half_range: float,
    period: float,
    noise_std: float,
    phase: float = 0.0,
) -> np.ndarray:
    """
    Return a length-N array:
        midpoint + half_range * sin(2π t / period + phase) + N(0, noise_std)

    The sinusoid provides smooth, continuous variation that mimics orbital
    or thermal cycles; noise_std adds sensor measurement noise on top.
    """
    wave  = half_range * np.sin(2 * np.pi * t / period + phase)
    noise = rng.normal(0.0, noise_std, N)
    return midpoint + wave + noise


# ─────────────────────────────────────────────────────────────────────────────
# 2. Normal baselines
#
#   Physical justification for each channel:
#
#   temperature        °C   — instrument bay; orbital cycle drives ±10 °C
#                             variation around a 15 °C mean
#   battery_voltage    V    — 28 V regulated bus; small ripple ±0.6 V
#   power_consumption  W    — nominal ops 50–150 W; slow activity cycle
#   radiation_level    mSv/h— interplanetary cruise baseline; low & stable
#   signal_strength    dBm  — good link margin −60 to −75 dBm
#   fuel_level         %    — monotonically decreasing with propulsion burns
#   solar_output       W    — 60–120 W; periodic dips for eclipse/night passes
# ─────────────────────────────────────────────────────────────────────────────

# 2a. Sinusoidal + noise channels
temperature       = smooth_signal(midpoint=15.0,  half_range=10.0, period=288.0,
                                  noise_std=1.2,  phase=0.0)
battery_voltage   = smooth_signal(midpoint=28.25, half_range=0.60, period=576.0,
                                  noise_std=0.08, phase=1.1)
power_consumption = smooth_signal(midpoint=100.0, half_range=25.0, period=192.0,
                                  noise_std=3.5,  phase=2.3)
radiation_level   = smooth_signal(midpoint=0.55,  half_range=0.20, period=400.0,
                                  noise_std=0.04, phase=0.7)
signal_strength   = smooth_signal(midpoint=-72.0, half_range=6.0,  period=350.0,
                                  noise_std=1.0,  phase=1.8)
solar_output      = smooth_signal(midpoint=90.0,  half_range=20.0, period=288.0,
                                  noise_std=2.0,  phase=3.2)

# 2b. Fuel: linear depletion (~28 % over the 3.5-day mission) + tiny noise
fuel_level = np.linspace(100.0, 72.0, N) + rng.normal(0.0, 0.3, N)

# Clip all channels to their physically plausible operating envelopes
temperature       = np.clip(temperature,       -25.0, 45.0)
battery_voltage   = np.clip(battery_voltage,    26.5, 29.8)
power_consumption = np.clip(power_consumption,  45.0, 160.0)
radiation_level   = np.clip(radiation_level,     0.1,  1.2)
signal_strength   = np.clip(signal_strength,   -92.0, -55.0)
fuel_level        = np.clip(fuel_level,          0.0, 100.0)
solar_output      = np.clip(solar_output,        50.0, 130.0)


# ─────────────────────────────────────────────────────────────────────────────
# 3. Anomaly event injection
#
#   Strategy: pick 6 non-overlapping start positions (one per anomaly type),
#   each spanning a short contiguous window of 3–8 readings. The values
#   inside each window are ramped to their anomalous range to simulate
#   a real fault that builds, peaks, and begins to recover.
#
#   Total anomalous rows ≈ 50 (5 % of 1,000).
#   Target per event: 8 rows  →  6 events × 8 rows = 48 rows  ≈ 5 %.
# ─────────────────────────────────────────────────────────────────────────────

# Each event definition: (anomaly_type, window_length)
EVENTS = [
    ("thermal",   8),   # temperature spike
    ("power",     8),   # power consumption surge
    ("battery",   8),   # battery voltage drop
    ("radiation", 8),   # radiation burst
    ("comms",     8),   # signal-strength fade
    ("solar",     8),   # solar output drop
]

# Place events with guaranteed separation of ≥30 rows (150 min) between them.
# Reserve the first and last 20 rows so anomalies don't clip the edges.
GAP     = 30
MARGIN  = 20

used = []   # list of (start, end) intervals already placed

def _find_start(win_len: int, max_attempts: int = 10_000) -> int:
    """Draw a random start index that doesn't overlap any placed event."""
    for _ in range(max_attempts):
        s = int(rng.integers(MARGIN, N - MARGIN - win_len))
        e = s + win_len
        # Reject if too close to any existing event
        if all(e + GAP <= es or s - GAP >= ee for es, ee in used):
            return s
    raise RuntimeError("Could not place anomaly event without overlap")

# Ramp helper: creates a smooth up-then-down profile over `length` steps
def _ramp(length: int) -> np.ndarray:
    """Triangular ramp: rises to peak at midpoint then falls back."""
    half = length / 2.0
    return np.array([1.0 - abs(i - half + 0.5) / half for i in range(length)])

event_log = []   # records (type, start_idx, end_idx, anomaly_values)

for atype, wlen in EVENTS:
    s = _find_start(wlen)
    e = s + wlen
    used.append((s, e))
    ramp = _ramp(wlen)   # smooth profile within the window

    if atype == "thermal":
        # Heater runaway / solar exposure — temperature climbs to 80-130 °C
        peak  = rng.uniform(85.0, 125.0)
        baseline_val = temperature[s:e].copy()
        temperature[s:e] = baseline_val + ramp * (peak - baseline_val)
        event_log.append((atype, s, e, temperature[s:e].round(2).tolist()))

    elif atype == "power":
        # Subsystem fault — power spikes to 250-400 W
        peak  = rng.uniform(260.0, 390.0)
        baseline_val = power_consumption[s:e].copy()
        power_consumption[s:e] = baseline_val + ramp * (peak - baseline_val)
        event_log.append((atype, s, e, power_consumption[s:e].round(2).tolist()))

    elif atype == "battery":
        # Cell failure — voltage drops to 18-23 V
        trough = rng.uniform(18.5, 23.0)
        baseline_val = battery_voltage[s:e].copy()
        battery_voltage[s:e] = baseline_val - ramp * (baseline_val - trough)
        event_log.append((atype, s, e, battery_voltage[s:e].round(2).tolist()))

    elif atype == "radiation":
        # Solar particle event — radiation climbs to 6-14 mSv/h
        peak  = rng.uniform(6.0, 14.0)
        baseline_val = radiation_level[s:e].copy()
        radiation_level[s:e] = baseline_val + ramp * (peak - baseline_val)
        event_log.append((atype, s, e, radiation_level[s:e].round(4).tolist()))

    elif atype == "comms":
        # Link degradation — signal drops to -115 to -130 dBm
        trough = rng.uniform(-128.0, -115.0)
        baseline_val = signal_strength[s:e].copy()
        signal_strength[s:e] = baseline_val - ramp * (baseline_val - trough)
        event_log.append((atype, s, e, signal_strength[s:e].round(2).tolist()))

    elif atype == "solar":
        # Panel occlusion — output drops to 3-15 W
        trough = rng.uniform(3.0, 15.0)
        baseline_val = solar_output[s:e].copy()
        solar_output[s:e] = baseline_val - ramp * (baseline_val - trough)
        event_log.append((atype, s, e, solar_output[s:e].round(2).tolist()))


# ─────────────────────────────────────────────────────────────────────────────
# 4. Assemble & save DataFrame
#    No label column — anomaly detection is performed by the ML model.
# ─────────────────────────────────────────────────────────────────────────────

timestamps = pd.date_range(start=START_TS, periods=N, freq=CADENCE)

df = pd.DataFrame({
    "timestamp":         timestamps,
    "temperature":       np.round(temperature,       2),
    "battery_voltage":   np.round(battery_voltage,   2),
    "power_consumption": np.round(power_consumption, 2),
    "radiation_level":   np.round(radiation_level,   4),
    "signal_strength":   np.round(signal_strength,   2),
    "fuel_level":        np.round(fuel_level,         2),
    "solar_output":      np.round(solar_output,       2),
})

os.makedirs("data", exist_ok=True)
df.to_csv(OUT_PATH, index=False)


# ─────────────────────────────────────────────────────────────────────────────
# 5. Console summary
# ─────────────────────────────────────────────────────────────────────────────

# Collect all anomaly indices to compute counts
all_anom_idx = set()
for _, s, e, _ in event_log:
    all_anom_idx.update(range(s, e))

total_anom = len(all_anom_idx)

print("AstraGuard AI -- Simulated Telemetry Generator (v2)")
print("=" * 60)
print(f"  Records written     : {len(df):,}")
print(f"  Anomalous records   : {total_anom} ({total_anom / N:.1%})")
print(f"  Anomaly events      : {len(event_log)}")
print(f"  Time span           : {df['timestamp'].iloc[0]}  to  {df['timestamp'].iloc[-1]}")
print(f"  Output              : {OUT_PATH}")

# Normal-range stats (exclude anomaly rows)
normal_mask = ~df.index.isin(all_anom_idx)
normal_df   = df[normal_mask]

print("\n--- Normal Operating Ranges (950 records) ---")
print(f"  {'Channel':<22} {'Min':>10} {'Mean':>10} {'Max':>10}  Unit")
print(f"  {'-'*60}")
UNITS = {
    "temperature": "degC", "battery_voltage": "V",
    "power_consumption": "W", "radiation_level": "mSv/h",
    "signal_strength": "dBm", "fuel_level": "%", "solar_output": "W",
}
for col in ["temperature","battery_voltage","power_consumption",
            "radiation_level","signal_strength","fuel_level","solar_output"]:
    s = normal_df[col]
    print(f"  {col:<22} {s.min():>10.3f} {s.mean():>10.3f} {s.max():>10.3f}  {UNITS[col]}")

print("\n--- Injected Anomaly Events ---")
for atype, s, e, vals in event_log:
    ts_start = df["timestamp"].iloc[s]
    ts_end   = df["timestamp"].iloc[e - 1]
    peak     = max(vals, key=abs) if atype in ("thermal","power","radiation") else min(vals)
    print(f"  {atype:<12}  rows {s:>4}-{e-1:<4}  "
          f"{str(ts_start)[:16]}  to  {str(ts_end)[:16]}  "
          f"  peak/trough = {peak}")

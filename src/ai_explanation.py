"""
src/ai_explanation.py
──────────────────────────────────────────────────────────────────────────────
AstraGuard AI — Mission Analysis & Natural-Language Explanation Engine
IBM AI Builders Challenge, August 2026: Advance Space Exploration with AI

PROTOTYPE DISCLAIMER
────────────────────
All analyses produced by this module are PROTOTYPE decision-support outputs
generated from SIMULATED spacecraft telemetry. They are NOT certified
aerospace assessments and must NOT be used for real mission operations or
safety-critical decisions.

PURPOSE
───────
Convert structured anomaly-detection and risk-scoring results into concise,
operator-readable natural-language mission analyses. The language deliberately
uses hedged phrasing ("may indicate", "possible", "warrants investigation")
because the underlying data is simulated and the model is a prototype.

DESIGN FOR AI INTEGRATION
──────────────────────────
The module is split into two clearly separated layers:

  Layer 1 — PROMPT BUILDER  (always runs, no external dependencies)
    build_prompt()  assembles a structured context + instruction string
    that fully describes the telemetry situation. This string is the
    single artifact that gets sent to any LLM.

  Layer 2 — EXPLANATION GENERATOR  (pluggable back-end)
    generate_explanation() has a `backend` parameter:

      "template"  (default) — pure-Python template renderer, no API key
                              needed; suitable for demo / offline use.

      "openai"              — calls OpenAI Chat Completions API.
                              Requires: pip install openai
                              Env var:  OPENAI_API_KEY

      "watsonx"             — calls IBM watsonx.ai text generation API.
                              Requires: pip install ibm-watsonx-ai
                              Env vars: WATSONX_API_KEY, WATSONX_PROJECT_ID
                                        WATSONX_URL (optional, defaults to
                                        https://us-south.ml.cloud.ibm.com)

      Any callable          — if `backend` is a Python callable it is called
                              as backend(prompt: str) -> str, allowing you to
                              plug in any model without changing this file.

USAGE
─────
    from src.ai_explanation import generate_explanation, build_prompt, ExplanationInput

    inp = ExplanationInput(
        timestamp        = "2026-01-02 13:20:00",
        telemetry        = {"temperature": 12.92, "battery_voltage": 21.41, ...},
        anomaly_detected = True,
        anomaly_score    = -0.137,
        risk_score       = 73,
        risk_level       = "HIGH",
        top_factors      = ["Battery voltage anomaly (battery_voltage: 21.41 V)"],
        contributions    = [...],   # list of ChannelContribution from risk_engine
    )

    text = generate_explanation(inp)                        # offline template
    text = generate_explanation(inp, backend="openai")      # GPT-4o
    text = generate_explanation(inp, backend="watsonx")     # IBM Granite
    text = generate_explanation(inp, backend=my_fn)         # any callable
"""

from __future__ import annotations

import os
import textwrap
from dataclasses import dataclass, field
from typing import Any, Callable

# ─────────────────────────────────────────────────────────────────────────────
# 1.  Input data contract
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ExplanationInput:
    """
    All information needed to generate one mission analysis paragraph.

    Fields map directly onto the outputs of risk_engine.assess_row() so the
    two modules compose naturally:

        assessment = assess_row(row, flag, score)
        inp = ExplanationInput.from_assessment(row, assessment)
        text = generate_explanation(inp)
    """
    timestamp:        str
    telemetry:        dict[str, float]    # raw channel values
    anomaly_detected: bool
    anomaly_score:    float               # IsolationForest decision score
    risk_score:       int                 # MRI [0, 100]
    risk_level:       str                 # LOW | MEDIUM | HIGH | CRITICAL
    top_factors:      list[str]           # human-readable factor strings
    # contributions is optional — used for richer template rendering
    contributions:    list[Any] = field(default_factory=list)

    @classmethod
    def from_assessment(
        cls,
        row: dict | Any,
        assessment: Any,
    ) -> "ExplanationInput":
        """
        Convenience constructor: build from a raw telemetry row and a
        RiskAssessment returned by risk_engine.assess_row().
        """
        # Accept both dict and pd.Series
        telem = {
            k: float(row[k])
            for k in ("temperature", "battery_voltage", "power_consumption",
                      "radiation_level", "signal_strength", "fuel_level",
                      "solar_output")
        }
        ts = str(row.get("timestamp", "unknown"))
        return cls(
            timestamp=ts,
            telemetry=telem,
            anomaly_detected=assessment.anomaly_detected,
            anomaly_score=assessment.anomaly_score,
            risk_score=assessment.risk_score,
            risk_level=assessment.risk_level,
            top_factors=assessment.top_factors,
            contributions=assessment.contributions,
        )


# ─────────────────────────────────────────────────────────────────────────────
# 2.  Channel metadata — plain descriptions used in both template and prompts
# ─────────────────────────────────────────────────────────────────────────────

# Maps channel name → (normal range description, what exceedance may indicate)
CHANNEL_CONTEXT: dict[str, tuple[str, str]] = {
    "temperature": (
        "normal range -5 to 30 °C",
        "possible thermal event such as heater runaway, solar overexposure, "
        "or cooling system failure",
    ),
    "battery_voltage": (
        "normal range 27.0–29.5 V on the 28 V regulated bus",
        "possible battery cell failure, deep discharge, or power bus fault "
        "that may lead to loss of power to critical subsystems",
    ),
    "power_consumption": (
        "normal range 55–145 W during nominal operations",
        "possible subsystem fault, short circuit, or unplanned activation of "
        "power-intensive equipment",
    ),
    "radiation_level": (
        "normal range 0.1–1.3 mSv/h during interplanetary cruise",
        "possible solar particle event, passage through a radiation belt, or "
        "sensor anomaly requiring instrument-dose assessment",
    ),
    "signal_strength": (
        "normal range -82 to -55 dBm indicating a healthy link margin",
        "possible antenna misalignment, atmospheric or plasma interference, "
        "or hardware degradation affecting communication reliability",
    ),
    "solar_output": (
        "normal range 55–125 W with solar arrays in full sunlight",
        "possible panel occlusion, surface degradation, attitude-control drift "
        "pointing panels away from the Sun, or eclipse entry",
    ),
    "fuel_level": (
        "depletes gradually over the mission",
        "unusually rapid depletion may indicate an unplanned thruster firing "
        "or propellant leak",
    ),
}

# Operator investigation actions keyed by channel
INVESTIGATION_ACTIONS: dict[str, str] = {
    "temperature":        "Check thermal control system status, heater circuits, "
                          "and verify attitude relative to solar direction.",
    "battery_voltage":    "Review battery state-of-charge telemetry, check cell "
                          "balancing data, and verify power bus load shedding.",
    "power_consumption":  "Identify which subsystem is drawing excess current via "
                          "individual subsystem power telemetry.",
    "radiation_level":    "Cross-check with solar-event monitoring; consider "
                          "enabling instrument safe-mode if levels persist.",
    "signal_strength":    "Verify antenna pointing angles, check transponder "
                          "health, and review uplink/downlink logs for dropouts.",
    "solar_output":       "Confirm solar array deployment and pointing; check for "
                          "surface contamination flags or eclipse schedule.",
    "fuel_level":         "Audit recent thruster firing logs and inspect propellant "
                          "system pressure readings for leak indicators.",
}

# Risk-level urgency phrases used in the opening sentence
URGENCY_PHRASE: dict[str, str] = {
    "LOW":      "Telemetry appears nominal.",
    "MEDIUM":   "Telemetry shows elevated indicators that warrant operator attention.",
    "HIGH":     "Telemetry indicates a significant anomaly requiring immediate investigation.",
    "CRITICAL": "Telemetry shows critical conditions that may pose a mission-threatening risk.",
}


# ─────────────────────────────────────────────────────────────────────────────
# 3.  Prompt builder  (Layer 1 — always available, no API needed)
# ─────────────────────────────────────────────────────────────────────────────

def build_prompt(inp: ExplanationInput) -> str:
    """
    Assemble a fully self-contained LLM prompt from an ExplanationInput.

    The prompt includes:
      • System role / context paragraph
      • Structured telemetry snapshot
      • Anomaly and risk summary
      • Contributing factors with per-channel context
      • Four specific questions the analysis must answer
      • Tone / hedging instructions

    Returns a single string ready to be sent to any LLM as the user message
    (or as the content of a user-role chat message).
    """
    # --- Telemetry snapshot block ---
    telem_lines = []
    for ch, val in inp.telemetry.items():
        ctx, _ = CHANNEL_CONTEXT.get(ch, ("", ""))
        telem_lines.append(f"  {ch:<22} = {val:>9.3f}   ({ctx})")
    telem_block = "\n".join(telem_lines)

    # --- Contributing factors block ---
    if inp.top_factors:
        factors_block = "\n".join(f"  • {f}" for f in inp.top_factors)
    else:
        factors_block = "  • No individual channel exceeded its threshold. " \
                        "The IsolationForest model detected a multivariate deviation."

    # --- Per-channel anomaly context (for contributing channels only) ---
    contrib_detail = []
    for c in inp.contributions:
        if getattr(c, "is_contributing", False):
            _, implication = CHANNEL_CONTEXT.get(c.channel, ("", "unknown channel"))
            contrib_detail.append(
                f"  {c.channel}: value {c.value} {c.unit} "
                f"(threshold exceedance score {c.threshold_score:.0f}/100) — "
                f"{implication}."
            )
    contrib_block = "\n".join(contrib_detail) if contrib_detail else "  (none exceeded individual thresholds)"

    # --- Assemble full prompt ---
    prompt = textwrap.dedent(f"""
        You are AstraGuard AI, a mission-support assistant for a simulated
        spacecraft telemetry monitoring system. You help operators understand
        anomalies detected in telemetry data.

        IMPORTANT: All data below is SIMULATED telemetry from an AI prototype.
        It is NOT real spacecraft data. Use cautious, hedged language such as
        "may indicate", "possible", "appears to", "warrants investigation".
        Never present findings as certain facts.

        [TELEMETRY SNAPSHOT]
        Timestamp : {inp.timestamp}

        {telem_block}

        [ANOMALY DETECTION]
        Anomaly detected by IsolationForest : {"YES" if inp.anomaly_detected else "NO"}
        IsolationForest decision score      : {inp.anomaly_score:.6f}
          (more negative = stronger anomaly; normal range ~0.0 to +0.12)

        [MISSION RISK INDEX]
        Risk Score  : {inp.risk_score} / 100
        Risk Level  : {inp.risk_level}
          (0-30 LOW | 31-60 MEDIUM | 61-80 HIGH | 81-100 CRITICAL)

        [CONTRIBUTING FACTORS]
        {factors_block}

        [CHANNEL-LEVEL EXCEEDANCE DETAIL]
        {contrib_block}

        [INSTRUCTIONS]
        Write a concise mission analysis (4–6 sentences, plain prose, no
        bullet points or headers) that answers all four questions below:

          1. What anomaly was detected, and in which telemetry channel(s)?
          2. Why might this matter to the mission (what could it indicate)?
          3. Which telemetry signals contributed most and how severely?
          4. What should the operator investigate or check next?

        Tone: professional, clear, hedged. Begin with the urgency summary:
        "{URGENCY_PHRASE[inp.risk_level]}"
    """).strip()

    return prompt


# ─────────────────────────────────────────────────────────────────────────────
# 4.  Template renderer  (offline fallback — no API key required)
# ─────────────────────────────────────────────────────────────────────────────

def _render_template(inp: ExplanationInput) -> str:
    """
    Generate a natural-language mission analysis using pure Python string
    templates. No external API required. Suitable for demos and offline use.

    The output mirrors the 4-question structure required by the LLM prompt
    so that swapping in a real LLM produces a visually identical result.
    """
    # ── Opening: urgency ──────────────────────────────────────────────────────
    opening = URGENCY_PHRASE[inp.risk_level]

    # ── Q1 — What anomaly was detected? ──────────────────────────────────────
    if inp.anomaly_detected:
        detection_str = (
            f"At {inp.timestamp}, the IsolationForest anomaly detector flagged "
            f"this telemetry reading as anomalous (decision score "
            f"{inp.anomaly_score:.4f})."
        )
    else:
        detection_str = (
            f"At {inp.timestamp}, no statistical anomaly was detected by the "
            "IsolationForest model."
        )

    # ── Q2 — Why might it matter? (per contributing channel) ─────────────────
    contributing = [c for c in inp.contributions if getattr(c, "is_contributing", False)]
    if contributing:
        implications = []
        for c in contributing[:3]:   # top 3 to keep text concise
            _, impl = CHANNEL_CONTEXT.get(c.channel, ("", f"anomalous {c.channel}"))
            implications.append(
                f"the {c.channel} reading of {c.value} {c.unit} "
                f"may indicate {impl}"
            )
        implication_str = (
            "This may be significant because "
            + "; and ".join(implications) + "."
        )
    elif inp.anomaly_detected:
        implication_str = (
            "No individual channel exceeded its fixed threshold, but the "
            "IsolationForest model detected a possible multivariate deviation "
            "that may warrant closer inspection of combined subsystem behaviour."
        )
    else:
        implication_str = (
            "All channels are within their expected operating envelopes and "
            "no anomalous behaviour has been identified."
        )

    # ── Q3 — Which signals contributed most? ─────────────────────────────────
    if contributing:
        top = contributing[0]
        severity = (
            "critically high" if top.threshold_score >= 80 else
            "significantly elevated" if top.threshold_score >= 50 else
            "moderately elevated"
        )
        signal_str = (
            f"The primary contributing signal is {top.channel} with a "
            f"threshold exceedance score of {top.threshold_score:.0f}/100 "
            f"({severity}), resulting in a Mission Risk Index of "
            f"{inp.risk_score}/100 ({inp.risk_level})."
        )
        if len(contributing) > 1:
            others = ", ".join(c.channel for c in contributing[1:3])
            signal_str += f" Secondary contributors include: {others}."
    else:
        signal_str = (
            f"No channels exceeded their individual thresholds. "
            f"The Mission Risk Index is {inp.risk_score}/100 ({inp.risk_level})."
        )

    # ── Q4 — What to investigate next? ───────────────────────────────────────
    if contributing:
        actions = []
        for c in contributing[:2]:   # top 2 actionable channels
            action = INVESTIGATION_ACTIONS.get(c.channel, f"Review {c.channel} subsystem logs.")
            actions.append(action)
        investigate_str = (
            "Recommended next steps: "
            + " Additionally, ".join(actions)
        )
    elif inp.anomaly_detected:
        investigate_str = (
            "Recommended next steps: review the full telemetry snapshot for "
            "subtle combined deviations across multiple channels, and compare "
            "against historical baseline patterns for this mission phase."
        )
    else:
        investigate_str = (
            "No immediate investigation is required. Continue standard "
            "telemetry monitoring according to nominal operations procedure."
        )

    # ── Combine into a single paragraph ──────────────────────────────────────
    paragraph = " ".join([
        opening,
        detection_str,
        implication_str,
        signal_str,
        investigate_str,
    ])

    # Append prototype disclaimer as a final sentence
    paragraph += (
        " [PROTOTYPE: This analysis is generated from simulated telemetry "
        "and is intended for demonstration purposes only.]"
    )

    return paragraph


# ─────────────────────────────────────────────────────────────────────────────
# 5.  AI back-ends  (Layer 2 — swappable LLM integrations)
# ─────────────────────────────────────────────────────────────────────────────

def _call_openai(prompt: str, model: str = "gpt-4o-mini") -> str:
    """
    Send prompt to OpenAI Chat Completions API.

    Requirements:
        pip install openai
        Environment variable: OPENAI_API_KEY
    """
    try:
        from openai import OpenAI                           # type: ignore
    except ImportError as exc:
        raise ImportError(
            "openai package not installed. Run: pip install openai"
        ) from exc

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "OPENAI_API_KEY environment variable is not set."
        )

    client = OpenAI(api_key=api_key)
    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are AstraGuard AI, a mission-support assistant for a "
                    "simulated spacecraft telemetry monitoring prototype. "
                    "Always use hedged language and note when data is simulated."
                ),
            },
            {"role": "user", "content": prompt},
        ],
        temperature=0.3,        # low temperature for consistent, factual tone
        max_tokens=400,
    )
    return response.choices[0].message.content.strip()


def _call_watsonx(prompt: str) -> str:
    """
    Send prompt to IBM watsonx.ai text generation API using IBM Granite.

    Requirements:
        pip install ibm-watsonx-ai
        Environment variables:
            WATSONX_API_KEY      — IBM Cloud API key
            WATSONX_PROJECT_ID   — watsonx.ai project ID
            WATSONX_URL          — (optional) service URL
    """
    try:
        from ibm_watsonx_ai import Credentials                      # type: ignore
        from ibm_watsonx_ai.foundation_models import ModelInference  # type: ignore
    except ImportError as exc:
        raise ImportError(
            "ibm-watsonx-ai package not installed. "
            "Run: pip install ibm-watsonx-ai"
        ) from exc

    api_key    = os.environ.get("WATSONX_API_KEY")
    project_id = os.environ.get("WATSONX_PROJECT_ID")
    url        = os.environ.get("WATSONX_URL", "https://us-south.ml.cloud.ibm.com")

    if not api_key:
        raise EnvironmentError("WATSONX_API_KEY environment variable is not set.")
    if not project_id:
        raise EnvironmentError("WATSONX_PROJECT_ID environment variable is not set.")

    credentials = Credentials(url=url, api_key=api_key)
    model = ModelInference(
        model_id="ibm/granite-3-8b-instruct",
        credentials=credentials,
        project_id=project_id,
        params={
            "max_new_tokens": 400,
            "temperature":    0.3,
            "decoding_method": "greedy",
        },
    )
    response = model.generate_text(prompt=prompt)
    return response.strip()


# ─────────────────────────────────────────────────────────────────────────────
# 6.  Main public function
# ─────────────────────────────────────────────────────────────────────────────

def generate_explanation(
    inp: ExplanationInput,
    backend: str | Callable[[str], str] = "template",
    **backend_kwargs: Any,
) -> str:
    """
    Generate a natural-language mission analysis for the given telemetry event.

    Parameters
    ----------
    inp            : ExplanationInput — all telemetry, anomaly, and risk data.

    backend        : str or callable — which explanation engine to use.
        "template"  (default) — offline Python template, no API required.
        "openai"              — OpenAI Chat Completions (needs OPENAI_API_KEY).
        "watsonx"             — IBM watsonx.ai Granite (needs WATSONX_API_KEY,
                                WATSONX_PROJECT_ID).
        any callable          — called as backend(prompt: str) -> str.

    **backend_kwargs : passed through to the selected backend (e.g.
                       model="gpt-4o" for the openai backend).

    Returns
    -------
    str — multi-sentence natural-language mission analysis.
    """
    if backend == "template":
        return _render_template(inp)

    # For all non-template backends, first build the prompt then call the API
    prompt = build_prompt(inp)

    if backend == "openai":
        model = backend_kwargs.get("model", "gpt-4o-mini")
        return _call_openai(prompt, model=model)

    if backend == "watsonx":
        return _call_watsonx(prompt)

    if callable(backend):
        return backend(prompt)

    raise ValueError(
        f"Unknown backend '{backend}'. "
        "Valid options: 'template', 'openai', 'watsonx', or any callable."
    )


# ─────────────────────────────────────────────────────────────────────────────
# 7.  Batch helper — annotate an entire risk DataFrame
# ─────────────────────────────────────────────────────────────────────────────

def explain_dataframe(
    risk_df: Any,        # pd.DataFrame with risk_score, risk_level, top_factors
    assessed_rows: list[Any],  # parallel list of RiskAssessment objects
    backend: str | Callable[[str], str] = "template",
    anomaly_only: bool = True,
) -> Any:
    """
    Add an 'ai_explanation' column to a risk-assessed DataFrame.

    Parameters
    ----------
    risk_df        : DataFrame returned by risk_engine.assess_dataframe().
    assessed_rows  : list of RiskAssessment objects (one per DataFrame row),
                     returned by a loop over risk_engine.assess_row() calls.
    backend        : explanation backend (same options as generate_explanation).
    anomaly_only   : if True (default), only generate explanations for rows
                     where anomaly_flag == -1, leaving others as empty string.
                     Set False to explain every row (slow with API backends).

    Returns
    -------
    The same DataFrame with an 'ai_explanation' column appended.
    """
    import pandas as pd

    explanations = []
    for i, (_, row) in enumerate(risk_df.iterrows()):
        assessment = assessed_rows[i]
        skip = anomaly_only and not assessment.anomaly_detected
        if skip:
            explanations.append("")
        else:
            inp = ExplanationInput.from_assessment(row, assessment)
            explanations.append(generate_explanation(inp, backend=backend))

    out = risk_df.copy()
    out["ai_explanation"] = explanations
    return out

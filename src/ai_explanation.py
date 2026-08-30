"""
AstraGuard AI — Natural-Language Explanation Engine

Prototype decision-support module for simulated spacecraft telemetry.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Callable, ClassVar


# ---------------------------------------------------------------------------
# Telemetry configuration
# ---------------------------------------------------------------------------

TELEMETRY_CHANNELS: tuple[str, ...] = (
    "temperature",
    "battery_voltage",
    "power_consumption",
    "radiation_level",
    "signal_strength",
    "fuel_level",
    "solar_output",
)

CHANNEL_UNITS: dict[str, str] = {
    "temperature": "°C",
    "battery_voltage": "V",
    "power_consumption": "W",
    "radiation_level": "mSv/h",
    "signal_strength": "dBm",
    "fuel_level": "%",
    "solar_output": "W",
}


# ---------------------------------------------------------------------------
# Explanation input
# ---------------------------------------------------------------------------

@dataclass
class ExplanationInput:
    """Snapshot of telemetry and its risk assessment."""

    timestamp: str
    telemetry: dict[str, float]
    anomaly_detected: bool
    anomaly_score: float
    risk_score: int
    risk_level: str
    top_factors: list[str] = field(default_factory=list)
    contributions: list[Any] = field(default_factory=list)

    _CHANNELS: ClassVar[tuple[str, ...]] = TELEMETRY_CHANNELS

    @classmethod
    def from_assessment(
        cls,
        row: Any,
        assessment: Any,
    ) -> "ExplanationInput":

        def get_value(key: str, default: float = 0.0) -> float:
            try:
                if hasattr(row, "get"):
                    value = row.get(key, default)
                elif hasattr(row, "index") and key in row.index:
                    value = row[key]
                else:
                    value = default

                if value is None:
                    return default

                return float(value)

            except (TypeError, ValueError, KeyError):
                return default

        telemetry = {
            channel: get_value(channel)
            for channel in TELEMETRY_CHANNELS
        }

        timestamp = ""

        try:
            if hasattr(row, "get"):
                timestamp = str(row.get("timestamp", "") or "")
            elif hasattr(row, "index") and "timestamp" in row.index:
                timestamp = str(row["timestamp"] or "")
        except Exception:
            timestamp = ""

        return cls(
            timestamp=timestamp,
            telemetry=telemetry,
            anomaly_detected=bool(
                getattr(assessment, "anomaly_detected", False)
            ),
            anomaly_score=float(
                getattr(assessment, "anomaly_score", 0.0)
            ),
            risk_score=int(
                getattr(assessment, "risk_score", 0)
            ),
            risk_level=str(
                getattr(assessment, "risk_level", "LOW")
            ).upper(),
            top_factors=list(
                getattr(assessment, "top_factors", []) or []
            ),
            contributions=list(
                getattr(assessment, "contributions", []) or []
            ),
        )


# ---------------------------------------------------------------------------
# Telemetry interpretation
# ---------------------------------------------------------------------------

CHANNEL_CONTEXT: dict[str, dict[str, str]] = {

    "temperature": {
        "normal": "normally operates between -5 °C and 30 °C",
        "implication": (
            "A temperature deviation may indicate a possible thermal-control "
            "issue, heater failure, overheating, or sensor fault."
        ),
    },

    "battery_voltage": {
        "normal": "normally operates near 28 V (27.0–29.5 V)",
        "implication": (
            "A voltage deviation may indicate possible battery degradation, "
            "excessive load, or charging-system issues."
        ),
    },

    "power_consumption": {
        "normal": "normally remains between 55 W and 145 W",
        "implication": (
            "Unexpected power consumption may indicate an abnormal subsystem "
            "load, electrical fault, or a subsystem becoming inactive."
        ),
    },

    "radiation_level": {
        "normal": "normally ranges from 0.10 to 1.30 mSv/h",
        "implication": (
            "Elevated radiation may indicate a possible solar particle event "
            "or passage through a higher-radiation region."
        ),
    },

    "signal_strength": {
        "normal": "normally falls between -82 dBm and -55 dBm",
        "implication": (
            "Weak signal strength may indicate possible antenna misalignment, "
            "obstruction, communication hardware issues, or occultation."
        ),
    },

    "fuel_level": {
        "normal": "depletes gradually according to the mission profile",
        "implication": (
            "Unexpected fuel depletion may indicate a possible leak or "
            "unplanned propulsion activity."
        ),
    },

    "solar_output": {
        "normal": "normally ranges from 55 W to 125 W when illuminated",
        "implication": (
            "Low solar output may indicate possible panel degradation, "
            "shadowing, orientation problems, or debris impact."
        ),
    },
}


# ---------------------------------------------------------------------------
# Investigation recommendations
# ---------------------------------------------------------------------------

INVESTIGATION_ACTIONS: dict[str, str] = {

    "temperature": (
        "Check thermal-control heater and cooler status. "
        "Review nearby temperature sensors and look for recent thermal events."
    ),

    "battery_voltage": (
        "Review cell-level voltage telemetry and recent charge/discharge cycles. "
        "Verify power bus load conditions and load-shedding thresholds."
    ),

    "power_consumption": (
        "Identify currently active subsystems and review power-distribution logs. "
        "Check for unexpected load changes in the minutes prior to this reading."
    ),

    "radiation_level": (
        "Compare the reading with space-weather alerts and radiation detector data. "
        "Consider enabling a protective operating mode if elevated levels persist."
    ),

    "signal_strength": (
        "Verify antenna pointing and check communication-system status. "
        "Review scheduled occultation periods and uplink/downlink logs."
    ),

    "fuel_level": (
        "Review propulsion-system valve and pressure telemetry. "
        "Compare consumption with the planned burn schedule and check for unexpected propulsion events."
    ),

    "solar_output": (
        "Check solar-array orientation and deployment status. "
        "Review eclipse periods and compare output against the expected generation curve."
    ),
}


# ---------------------------------------------------------------------------
# Risk language
# ---------------------------------------------------------------------------

URGENCY_PHRASE: dict[str, str] = {
    "LOW": (
        "Telemetry appears to remain within generally nominal "
        "operating conditions."
    ),
    "MEDIUM": (
        "Telemetry shows elevated indicators that warrant operator attention."
    ),
    "HIGH": (
        "Telemetry indicates a significant anomaly requiring prompt "
        "investigation."
    ),
    "CRITICAL": (
        "Telemetry suggests a potentially serious condition requiring "
        "immediate review."
    ),
}


# ---------------------------------------------------------------------------
# Prompt builder for optional LLM backends
# ---------------------------------------------------------------------------

def build_prompt(inp: ExplanationInput) -> str:

    telemetry_lines = "\n".join(
        f"{channel}: "
        f"{inp.telemetry.get(channel, 0.0):.3f} "
        f"{CHANNEL_UNITS.get(channel, '')}"
        for channel in TELEMETRY_CHANNELS
    )

    if inp.top_factors:
        factors = ", ".join(inp.top_factors[:4])
    else:
        factors = "No individual threshold factor was identified."

    contribution_lines = []

    for contribution in inp.contributions:
        if getattr(contribution, "is_contributing", False):
            channel = getattr(contribution, "channel", "unknown")
            value = getattr(contribution, "value", "N/A")
            score = getattr(contribution, "threshold_score", 0)

            contribution_lines.append(
                f"{channel}: value={value}, severity={score}/100"
            )

    contributions = (
        "\n".join(contribution_lines)
        if contribution_lines
        else "No individual threshold exceedance."
    )

    return f"""
You are AstraGuard AI, a mission-support assistant for a simulated
spacecraft telemetry monitoring system.

IMPORTANT:
All telemetry is SIMULATED.
This is a prototype demonstration.
Do not present conclusions as certain aerospace findings.

Use cautious language such as:
"may indicate", "possible", "appears to", and "warrants investigation".

Telemetry:
{telemetry_lines}

Timestamp:
{inp.timestamp or "N/A"}

Anomaly detected:
{"YES" if inp.anomaly_detected else "NO"}

IsolationForest anomaly score:
{inp.anomaly_score:.6f}

Mission Risk Index:
{inp.risk_score}/100

Risk level:
{inp.risk_level}

Top risk factors:
{factors}

Contributing telemetry signals:
{contributions}

Write a concise 4–6 sentence mission analysis.

Explain:
1. What anomaly was detected.
2. Which telemetry channel contributed.
3. Why it may matter.
4. What the operator should investigate next.

Begin with:
"{URGENCY_PHRASE.get(inp.risk_level, URGENCY_PHRASE['LOW'])}"
""".strip()


# ---------------------------------------------------------------------------
# Offline AI-style explanation engine
# ---------------------------------------------------------------------------

def _render_template(inp: ExplanationInput) -> str:

    risk_level = inp.risk_level.upper()

    # Find contributing channels.
    contributing = [
        contribution
        for contribution in inp.contributions
        if getattr(contribution, "is_contributing", False)
    ]

    # Sort strongest contributors first.
    contributing.sort(
        key=lambda item: getattr(item, "threshold_score", 0),
        reverse=True,
    )

    primary = contributing[0] if contributing else None

    primary_channel = (
        getattr(primary, "channel", None)
        if primary
        else None
    )

    # ---------------------------------------------------------------
    # Sentence 1 — anomaly status
    # ---------------------------------------------------------------

    if inp.anomaly_detected:

        opening = (
            f"{URGENCY_PHRASE.get(risk_level, URGENCY_PHRASE['LOW'])} "
            f"The IsolationForest model flagged this telemetry reading "
            f"as a possible anomaly with a decision score of "
            f"{inp.anomaly_score:.4f}."
        )

    else:

        opening = (
            f"{URGENCY_PHRASE.get(risk_level, URGENCY_PHRASE['LOW'])} "
            f"The IsolationForest model did not flag this reading as "
            f"an anomaly, with a decision score of "
            f"{inp.anomaly_score:.4f}."
        )

    # ---------------------------------------------------------------
    # Sentence 2 — primary telemetry signal
    # ---------------------------------------------------------------

    if primary_channel in CHANNEL_CONTEXT:

        value = getattr(primary, "value", "N/A")
        unit = getattr(primary, "unit", "")

        context = CHANNEL_CONTEXT[primary_channel]

        channel_sentence = (
            f"The strongest contributing signal appears to be "
            f"{primary_channel.replace('_', ' ')} at "
            f"{value} {unit}, compared with a channel that "
            f"{context['normal']}."
        )

    elif inp.top_factors:

        factors = ", ".join(inp.top_factors[:3])

        channel_sentence = (
            f"The analysis identified the following risk factors: "
            f"{factors}."
        )

    else:

        channel_sentence = (
            "No single telemetry channel clearly exceeded its "
            "individual threshold, so the anomaly may reflect a "
            "multivariate deviation across several signals."
        )

    # ---------------------------------------------------------------
    # Sentence 3 — why it matters
    # ---------------------------------------------------------------

    if primary_channel in CHANNEL_CONTEXT:

        implication = CHANNEL_CONTEXT[primary_channel]["implication"]

    else:

        implication = (
            "The combined deviation may indicate an emerging subsystem "
            "condition, although additional telemetry review would be "
            "needed to determine the cause."
        )

    impact_sentence = implication

    # ---------------------------------------------------------------
    # Sentence 4 — other contributors
    # ---------------------------------------------------------------

    if len(contributing) > 1:

        additional = [
            getattr(item, "channel", "unknown").replace("_", " ")
            for item in contributing[1:3]
        ]

        contributor_sentence = (
            "Additional contributing signals include "
            + ", ".join(additional)
            + "."
        )

    else:

        contributor_sentence = (
            "No additional individually significant channel "
            "contributions were identified."
        )

    # ---------------------------------------------------------------
    # Sentence 5 — risk
    # ---------------------------------------------------------------

    risk_sentence = (
        f"The Mission Risk Index is {inp.risk_score}/100 "
        f"({risk_level}), representing the prototype's current "
        f"risk classification."
    )

    # ---------------------------------------------------------------
    # Sentence 6 — recommended investigation
    # ---------------------------------------------------------------

    if primary_channel in INVESTIGATION_ACTIONS:

        investigation_sentence = (
            "Recommended next steps: "
            + INVESTIGATION_ACTIONS[primary_channel]
        )

    else:

        investigation_sentence = (
            "Recommended next step: review telemetry around this "
            "timestamp and check for concurrent subsystem events."
        )

    # ---------------------------------------------------------------
    # Prototype disclaimer
    # ---------------------------------------------------------------

    disclaimer = (
        " Prototype disclaimer: this analysis is generated from "
        "simulated telemetry and is intended for demonstration and "
        "decision-support purposes only."
    )

    return (
        f"{opening} "
        f"{channel_sentence} "
        f"{impact_sentence} "
        f"{contributor_sentence} "
        f"{risk_sentence} "
        f"{investigation_sentence}"
        f"{disclaimer}"
    )


# ---------------------------------------------------------------------------
# Optional OpenAI backend
# ---------------------------------------------------------------------------

def _call_openai(
    prompt: str,
    model: str = "gpt-4o-mini",
) -> str:

    try:
        import openai
    except ImportError as exc:
        raise ImportError(
            "Install OpenAI with: pip install openai"
        ) from exc

    api_key = os.environ.get("OPENAI_API_KEY")

    if not api_key:
        raise EnvironmentError(
            "OPENAI_API_KEY environment variable is not set."
        )

    client = openai.OpenAI(api_key=api_key)

    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are AstraGuard AI, a prototype spacecraft "
                    "telemetry explanation assistant. All data is "
                    "simulated. Use cautious language and never claim "
                    "certainty."
                ),
            },
            {
                "role": "user",
                "content": prompt,
            },
        ],
        max_tokens=300,
        temperature=0.4,
    )

    return response.choices[0].message.content.strip()


# ---------------------------------------------------------------------------
# Optional IBM watsonx backend
# ---------------------------------------------------------------------------

def _call_watsonx(prompt: str) -> str:

    try:
        from ibm_watsonx_ai import APIClient, Credentials
        from ibm_watsonx_ai.foundation_models import ModelInference
        from ibm_watsonx_ai.metanames import (
            GenTextParamsMetaNames as GP,
        )
    except ImportError as exc:
        raise ImportError(
            "Install watsonx with: pip install ibm-watsonx-ai"
        ) from exc

    api_key = os.environ.get("WATSONX_API_KEY")

    if not api_key:
        raise EnvironmentError(
            "WATSONX_API_KEY environment variable is not set."
        )

    project_id = os.environ.get("WATSONX_PROJECT_ID")

    if not project_id:
        raise EnvironmentError(
            "WATSONX_PROJECT_ID environment variable is not set."
        )

    url = os.environ.get(
        "WATSONX_URL",
        "https://us-south.ml.cloud.ibm.com",
    )

    credentials = Credentials(
        url=url,
        api_key=api_key,
    )

    client = APIClient(credentials)

    model = ModelInference(
        model_id="ibm/granite-3-8b-instruct",
        api_client=client,
        project_id=project_id,
        params={
            GP.MAX_NEW_TOKENS: 300,
            GP.TEMPERATURE: 0.4,
        },
    )

    response = model.generate_text(
        prompt=prompt
    )

    return str(response).strip()


# ---------------------------------------------------------------------------
# Main explanation function
# ---------------------------------------------------------------------------

def generate_explanation(
    inp: ExplanationInput,
    backend: str | Callable[..., str] = "template",
    **backend_kwargs: Any,
) -> str:

    # Custom callable backend.
    if callable(backend) and not isinstance(backend, str):

        prompt = build_prompt(inp)

        return backend(
            prompt,
            **backend_kwargs,
        )

    # Default offline backend.
    if backend == "template":

        return _render_template(inp)

    # OpenAI backend.
    if backend == "openai":

        prompt = build_prompt(inp)

        model = backend_kwargs.pop(
            "model",
            "gpt-4o-mini",
        )

        return _call_openai(
            prompt,
            model=model,
        )

    # IBM watsonx backend.
    if backend == "watsonx":

        prompt = build_prompt(inp)

        return _call_watsonx(prompt)

    raise ValueError(
        f"Unknown backend: {backend!r}. "
        "Supported backends are: template, openai, watsonx, "
        "or a custom callable."
    )


# ---------------------------------------------------------------------------
# DataFrame helper
# ---------------------------------------------------------------------------

def explain_dataframe(
    risk_df: Any,
    assessed_rows: list[Any],
    backend: str | Callable[..., str] = "template",
    anomaly_only: bool = True,
) -> Any:

    if len(risk_df) != len(assessed_rows):

        raise ValueError(
            f"risk_df has {len(risk_df)} rows but assessed_rows has "
            f"{len(assessed_rows)} assessments."
        )

    explanations: list[str] = []

    for index, (_, row) in enumerate(risk_df.iterrows()):

        assessment = assessed_rows[index]

        if (
            anomaly_only
            and not getattr(
                assessment,
                "anomaly_detected",
                False,
            )
        ):

            explanations.append("")
            continue

        explanation_input = (
            ExplanationInput.from_assessment(
                row,
                assessment,
            )
        )

        explanation = generate_explanation(
            explanation_input,
            backend=backend,
        )

        explanations.append(explanation)

    result = risk_df.copy()

    result["ai_explanation"] = explanations

    return result
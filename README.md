# 🛰️ AstraGuard AI

## Space Mission Risk & Anomaly Intelligence

**AI-powered telemetry intelligence for spacecraft anomaly detection, mission-risk assessment, and AI-assisted analysis.**

**GitHub:** `https://github.com/soujanya04122007/astraguard-ai`

---

## 🌌 Overview

**AstraGuard AI** is an AI-powered prototype designed to monitor simulated spacecraft telemetry, detect abnormal behavior, evaluate mission risk, and transform complex telemetry events into understandable mission intelligence.

The platform combines **machine learning, telemetry analytics, risk scoring, interactive visualization, emergency simulation, and AI-assisted mission analysis** in a unified Streamlit dashboard.

### Core Pipeline

```text
🛰️ Telemetry
      ↓
📊 Data Processing
      ↓
🤖 Anomaly Detection
      ↓
⚠️ Risk Assessment
      ↓
🧠 AI Mission Analysis
      ↓
🚨 Recommended Investigation
```

> ⚠️ **PROTOTYPE:** AstraGuard AI uses simulated telemetry for demonstration purposes. It is not an official space-agency system and must not be used for operational or safety-critical decisions.

---

# 🎯 Problem Statement

Space missions continuously generate telemetry from multiple spacecraft subsystems including power, thermal, communications, radiation, and solar systems.

Monitoring these signals manually can make it difficult to identify unusual patterns quickly, especially when several signals change simultaneously.

### Challenges

* Large volumes of telemetry data
* Difficulty identifying abnormal patterns
* Multiple simultaneous subsystem changes
* Delayed interpretation of unusual readings
* Lack of a unified mission-risk view
* Need for faster anomaly investigation

### Our Solution

AstraGuard AI provides a single intelligent dashboard that:

1. Monitors simulated spacecraft telemetry.
2. Detects potentially anomalous observations.
3. Identifies important contributing signals.
4. Calculates a Mission Risk Index.
5. Explains detected events in human-readable language.
6. Provides recommended investigation steps.
7. Simulates complex emergency scenarios.

---

# 🧠 AI & Machine Learning Approach

## Isolation Forest

AstraGuard AI uses the **Isolation Forest** algorithm from `scikit-learn` for unsupervised anomaly detection.

Isolation Forest is useful for identifying observations that behave differently from the normal telemetry population without requiring every anomaly to be manually labelled.

### Detection Pipeline

```text
Telemetry Dataset
       ↓
Data Cleaning
       ↓
Feature Preparation
       ↓
Isolation Forest
       ↓
Anomaly Detection
       ↓
Severity Evaluation
```

The system then combines anomaly information with telemetry severity indicators to produce a mission-level risk assessment.

---

# 📊 Mission Risk Index

AstraGuard AI converts telemetry conditions into a **Mission Risk Index from 0–100**.

|  Score | Risk Level  | Meaning                            |
| -----: | ----------- | ---------------------------------- |
|   0–29 | 🟢 LOW      | Normal / low concern               |
|  30–59 | 🟡 MEDIUM   | Increased monitoring               |
|  60–79 | 🟠 HIGH     | Significant investigation required |
| 80–100 | 🔴 CRITICAL | Severe simulated condition         |

The risk score is designed to provide a quick, understandable representation of the simulated mission state.

---

# 🤖 AI Mission Analysis

When an anomaly is detected, AstraGuard AI generates an AI-assisted analysis containing:

* Anomaly timestamp
* Telemetry signal
* Observed value
* Detection result
* Severity / threshold exceedance
* Mission Risk Index
* Possible technical interpretation
* Recommended investigation steps

### Example Simulated Event

```text
Signal:        battery_voltage
Observed:      21.41 V
Risk Score:    73 / 100
Risk Level:    HIGH
Detector:      Isolation Forest
```

The purpose is to demonstrate how raw telemetry can be transformed into meaningful mission intelligence.

---

# 🚨 Emergency Simulation

AstraGuard AI includes an interactive **Simulate Emergency Event** feature.

The simulator injects a simultaneous multi-fault telemetry event involving:

* 🌡️ Thermal spike
* 🔋 Battery drop
* ⚡ Power surge
* ☢️ Radiation burst
* 📡 Communications fade
* ☀️ Solar-generation drop

The dashboard demonstrates the complete simulated response:

```text
NORMAL
  ↓
ANOMALY
  ↓
HIGH RISK
  ↓
AI ANALYSIS
  ↓
INVESTIGATION
```

This demonstrates how the platform can handle a complex multi-signal scenario rather than only a single threshold violation.

---

# 🖥️ Dashboard Features

## 01 — Mission Overview

Provides an at-a-glance view of:

* Mission status
* Spacecraft health
* Mission Risk Score
* Risk level
* Active anomalies
* Latest telemetry

## 02 — Telemetry Monitoring

Visualizes important simulated spacecraft telemetry signals.

## 03 — Active Anomalies

Highlights detected abnormal events and their severity.

## 04 — Mission Risk

Displays the calculated Mission Risk Index and current risk level.

## 05 — AI Mission Analysis

Converts anomaly information into a human-readable technical explanation.

## 06 — Emergency Simulation

Injects a simulated multi-fault event and demonstrates the complete AI pipeline.

## 07 — Ask AstraGuard

Provides an interactive interface for asking questions about the mission intelligence displayed by the prototype.

---

# 🏗️ System Architecture

```text
                 ┌─────────────────────┐
                 │ Simulated Telemetry │
                 │      telemetry.csv  │
                 └──────────┬──────────┘
                            │
                            ▼
                 ┌─────────────────────┐
                 │ Data Processing &   │
                 │ Feature Preparation │
                 └──────────┬──────────┘
                            │
                            ▼
                 ┌─────────────────────┐
                 │   Isolation Forest  │
                 │ Anomaly Detection   │
                 └──────────┬──────────┘
                            │
                            ▼
                 ┌─────────────────────┐
                 │ Severity & Anomaly  │
                 │     Assessment      │
                 └──────────┬──────────┘
                            │
                            ▼
                 ┌─────────────────────┐
                 │  Mission Risk Index │
                 │       0–100         │
                 └──────────┬──────────┘
                            │
                ┌───────────┴───────────┐
                ▼                       ▼
       ┌─────────────────┐     ┌─────────────────┐
       │ Streamlit       │     │ AI Mission      │
       │ Dashboard       │     │ Analysis        │
       └─────────────────┘     └─────────────────┘
                │                       │
                └───────────┬───────────┘
                            ▼
                 ┌─────────────────────┐
                 │ Mission Intelligence│
                 │     Dashboard       │
                 └─────────────────────┘
```

---

# 🛠️ Technology Stack

| Technology                    | Purpose                                     |
| ----------------------------- | ------------------------------------------- |
| **Python**                    | Application logic and data processing       |
| **Streamlit**                 | Interactive web dashboard                   |
| **Pandas**                    | Telemetry data processing                   |
| **NumPy**                     | Numerical operations                        |
| **scikit-learn**              | Isolation Forest anomaly detection          |
| **Plotly / Streamlit Charts** | Data visualization                          |
| **IBM Bob**                   | AI-assisted primary development environment |
| **GitHub**                    | Version control and project repository      |

---

# 📁 Project Structure

```text
astraguard-ai/
│
├── app.py
│       └── Main Streamlit application
│
├── telemetry.csv
│       └── Simulated spacecraft telemetry dataset
│
├── README.md
│       └── Project documentation
│
├── requirements.txt
│       └── Python dependencies
│
└── assets/
        └── Images, icons and visual resources
```

### Main Files

### `app.py`

Contains the main AstraGuard AI Streamlit application, including dashboard components, telemetry processing, anomaly detection, risk assessment, simulation, and analysis.

### `telemetry.csv`

Contains the simulated telemetry data used by the prototype.

### `README.md`

Contains project documentation, architecture, setup instructions, AI methodology, and challenge information.

### `requirements.txt`

Lists the Python packages required to run AstraGuard AI.

### `assets/`

Contains supporting visual resources used by the application.

---

# ⚙️ Installation & Setup

## 1. Clone the Repository

```bash
git clone https://github.com/soujanya04122007/astraguard-ai.git
cd astraguard-ai
```

## 2. Install Dependencies

```bash
pip install -r requirements.txt
```

If `requirements.txt` is not available, install the main dependencies:

```bash
pip install streamlit pandas numpy scikit-learn plotly
```

## 3. Run the Application

```bash
python -m streamlit run app.py
```

The application will be available at:

```text
http://localhost:8501
```

---

# 🧪 Demonstration

The current prototype demonstrates a simulated battery-voltage anomaly.

### Example

```text
Mission Status:       ALERT
Spacecraft Health:   27%
Mission Risk Score:  73 / 100
Risk Level:          HIGH

Anomaly:
Signal:              battery_voltage
Observed Value:      21.41 V
Detector:            Isolation Forest
```

The dashboard then provides AI-assisted mission analysis and recommended investigation steps.

---

# 🧩 Emergency Demonstration

To demonstrate the full pipeline:

### Step 1

Launch AstraGuard AI.

### Step 2

Review the current telemetry and mission status.

### Step 3

Click:

**Simulate Emergency Event**

### Step 4

Observe the simulated telemetry changes.

### Step 5

Observe the transition:

```text
NORMAL → ANOMALY → HIGH RISK → AI ANALYSIS
```

This provides judges with a clear demonstration of the system's end-to-end functionality.

---

# 🏆 IBM AI Builders Challenge

### Challenge

**IBM AI Builders Challenge — August 2026**

### Theme

**Advance Space Exploration with AI**

AstraGuard AI explores how AI and machine learning can support spacecraft telemetry intelligence and anomaly investigation.

### IBM Bob Usage

IBM Bob was used as the primary AI-assisted development environment during the project.

It supported:

* Application development
* Code generation
* Debugging
* Code refinement
* Dashboard development
* Problem solving
* Project iteration

The project demonstrates how AI-assisted development can accelerate the creation of an interactive machine-learning application.

---

# 🌍 Potential Impact

AstraGuard AI is currently a simulated prototype, but the underlying concept could be extended toward future telemetry-intelligence systems.

Potential applications include:

* 🛰️ Spacecraft health monitoring
* 📡 Satellite telemetry analysis
* ⚠️ Early anomaly identification
* 📊 Mission-risk visualization
* 🤖 Automated technical reporting
* 🔍 Multi-sensor anomaly correlation
* 🚨 Intelligent alert prioritization
* 🧑‍🚀 Operator decision support

### Vision

> **Turn massive streams of spacecraft telemetry into clear, explainable mission intelligence.**

---

# 🔮 Future Roadmap

Future versions of AstraGuard AI could introduce:

* Real-time telemetry streaming
* Advanced time-series anomaly detection
* LSTM / Transformer-based forecasting
* Automated anomaly classification
* Multi-sensor anomaly correlation
* Explainable AI
* Historical mission comparison
* Real satellite telemetry integration
* Automated alert notifications
* Digital-twin simulation
* Fault-propagation modeling
* Mission-control role-based interfaces

---

# 🔐 Safety & Data Disclaimer

AstraGuard AI is a **research and demonstration prototype**.

* All telemetry used in this project is simulated.
* This is not an official space-agency system.
* AI-generated analysis is for demonstration purposes.
* Risk scores are prototype outputs.
* The system must not be used for real spacecraft control.
* The system must not be used for safety-critical decisions.

---

# 👩‍💻 Project Information

**Project:** AstraGuard AI
**Domain:** Artificial Intelligence & Space Exploration
**Focus:** Telemetry Anomaly Detection & Mission Risk Intelligence
**Machine Learning:** Isolation Forest
**Dashboard:** Streamlit
**Development:** IBM Bob + VS Code
**Repository:** `https://github.com/soujanya04122007/astraguard-ai`

---

# ⭐ Key Innovation

AstraGuard AI does not stop at detecting an unusual telemetry value.

It connects multiple stages into one intelligence pipeline:

> **Detect → Understand → Quantify Risk → Explain → Investigate**

This makes the prototype more than a telemetry dashboard—it demonstrates an **AI-assisted mission intelligence workflow**.

---

## 🚀 AstraGuard AI

### **Detect anomalies. Understand risk. Accelerate mission intelligence.**

**Built for the IBM AI Builders Challenge — August 2026**

🛰️ **All telemetry is simulated. Prototype only.**

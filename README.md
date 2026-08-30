# 🛰️ AstraGuard AI

## Space Mission Risk & Anomaly Intelligence

**AstraGuard AI** is an AI-powered prototype for monitoring simulated spacecraft telemetry, detecting anomalies, assessing mission risk, and generating AI-assisted mission analysis through an interactive dashboard.

> ⚠️ **Prototype Disclaimer:** AstraGuard AI uses simulated telemetry for demonstration purposes only. It is not an official space-agency system and must not be used for operational or safety-critical decisions.

**GitHub:** `https://github.com/soujanya04122007/astraguard-ai`

---

# 🎯 Problem Statement

Spacecraft continuously generate large amounts of telemetry from critical subsystems such as power, thermal control, communications, radiation monitoring, and solar generation.

Manually monitoring these signals can make it difficult to identify abnormal behavior quickly, particularly when multiple telemetry values change simultaneously.

AstraGuard AI addresses this challenge by providing an intelligent pipeline that transforms raw telemetry into understandable mission intelligence.

### Key Challenges

* Large volumes of telemetry data
* Difficulty identifying abnormal patterns
* Multiple simultaneous subsystem changes
* Delayed anomaly interpretation
* Lack of a unified mission-risk view
* Need for faster investigation and prioritization

---

# 💡 Solution

AstraGuard AI combines **machine learning, telemetry analytics, risk scoring, visualization, emergency simulation, and AI-assisted analysis** into a single Streamlit application.

The system follows this pipeline:

```text
Telemetry Data
      ↓
Data Processing
      ↓
Machine Learning
      ↓
Anomaly Detection
      ↓
Risk Assessment
      ↓
AI Mission Analysis
      ↓
Investigation Support
```

Instead of simply displaying raw telemetry values, the system identifies unusual observations and converts them into a mission-level risk perspective.

---

# 🚀 Key Features

### 🛰️ 1. Mission Overview

Provides a centralized view of:

* Mission status
* Spacecraft health
* Mission Risk Score
* Risk level
* Active anomalies
* Latest telemetry timestamp

### 📊 2. Telemetry Monitoring

Displays important spacecraft telemetry signals through interactive visualizations.

### ⚠️ 3. Anomaly Detection

Uses an **Isolation Forest** machine-learning model to identify unusual telemetry observations.

### 📈 4. Mission Risk Scoring

Converts telemetry conditions and anomaly severity into a **Mission Risk Index from 0–100**.

### 🤖 5. AI Mission Analysis

Generates a human-readable explanation of detected anomalies, including:

* Anomaly timestamp
* Telemetry signal
* Observed value
* Detection result
* Severity
* Risk score
* Possible interpretation
* Recommended investigation steps

### 🚨 6. Emergency Simulation

The application can inject a simulated multi-fault event involving:

* Thermal spike
* Battery drop
* Power surge
* Radiation burst
* Communications fade
* Solar-generation drop

This demonstrates the complete pipeline:

```text
NORMAL
  ↓
ANOMALY
  ↓
HIGH RISK
  ↓
AI ANALYSIS
```

### 💬 7. Ask AstraGuard

Provides an interactive interface for asking questions about the mission intelligence presented by the prototype.

---

# 🧠 AI Approach

## Isolation Forest Anomaly Detection

AstraGuard AI uses **Isolation Forest** from `scikit-learn` for unsupervised anomaly detection.

The model identifies telemetry observations that differ significantly from the expected data distribution.

### AI Pipeline

```text
Simulated Telemetry
        ↓
Data Preparation
        ↓
Feature Processing
        ↓
Isolation Forest
        ↓
Anomaly Detection
        ↓
Severity Assessment
        ↓
Mission Risk Index
        ↓
AI Mission Analysis
```

The approach allows the prototype to detect unusual telemetry without requiring every possible anomaly to be manually labelled.

---

# 📐 System Architecture

```text
                  ┌──────────────────────┐
                  │  Simulated Telemetry │
                  │     telemetry.csv    │
                  └──────────┬───────────┘
                             │
                             ▼
                  ┌──────────────────────┐
                  │ Data Processing &    │
                  │ Feature Preparation  │
                  └──────────┬───────────┘
                             │
                             ▼
                  ┌──────────────────────┐
                  │   Isolation Forest   │
                  │  Anomaly Detection   │
                  └──────────┬───────────┘
                             │
                             ▼
                  ┌──────────────────────┐
                  │ Severity Assessment  │
                  │ & Risk Calculation   │
                  └──────────┬───────────┘
                             │
                             ▼
                  ┌──────────────────────┐
                  │  Mission Risk Index  │
                  │       0 – 100        │
                  └──────────┬───────────┘
                             │
                    ┌────────┴────────┐
                    ▼                 ▼
           ┌────────────────┐  ┌────────────────┐
           │   Streamlit    │  │ AI Mission     │
           │   Dashboard    │  │ Analysis       │
           └────────────────┘  └────────────────┘
                    │                 │
                    └────────┬────────┘
                             ▼
                  ┌──────────────────────┐
                  │ Mission Intelligence │
                  │      Interface       │
                  └──────────────────────┘
```

---

# 🛠️ Technology Stack

| Technology                    | Purpose                               |
| ----------------------------- | ------------------------------------- |
| **Python**                    | Application logic and data processing |
| **Streamlit**                 | Interactive dashboard                 |
| **Pandas**                    | Telemetry data processing             |
| **NumPy**                     | Numerical computation                 |
| **scikit-learn**              | Isolation Forest anomaly detection    |
| **Plotly / Streamlit Charts** | Telemetry visualization               |
| **IBM Bob**                   | AI-assisted development               |
| **Git**                       | Version control                       |
| **GitHub**                    | Source-code hosting                   |

---

# 📊 Dataset

AstraGuard AI uses **simulated spacecraft telemetry**.

The dataset is stored in:

```text
data/telemetry.csv
```

The telemetry represents measurements from spacecraft-related systems, including signals such as:

* Battery voltage
* Temperature
* Power
* Radiation
* Communications
* Solar generation

The project also generates processed outputs:

```text
data/anomaly_results.csv
data/risk_results.csv
```

These files contain the results produced by the anomaly-detection and risk-assessment pipeline.

> The dataset is simulated and does not contain official or operational spacecraft telemetry.

---

# 🤖 How IBM Bob Was Used

**IBM Bob was used as the primary AI-assisted development environment for AstraGuard AI.**

Bob supported the project across multiple stages of development:

### 1. Generate Telemetry-Processing Code

Bob helped generate and refine Python code for:

* Loading telemetry data
* Processing CSV data
* Preparing features
* Handling telemetry records
* Creating processed outputs

### 2. Develop the ML Pipeline

Bob assisted in structuring the machine-learning workflow from telemetry preprocessing through anomaly detection and risk assessment.

### 3. Implement Anomaly Detection

Bob helped implement and refine the **Isolation Forest** anomaly-detection component using `scikit-learn`.

### 4. Debug Python / Streamlit Code

Bob was used to identify and resolve:

* Python runtime errors
* Streamlit execution errors
* Data-processing issues
* Function and variable errors
* Application integration problems

### 5. Build Dashboard Components

Bob helped develop Streamlit dashboard components for:

* Mission overview
* Telemetry monitoring
* Active anomalies
* Mission risk
* AI mission analysis
* Emergency simulation
* Ask AstraGuard

### 6. Improve UI

Bob supported improvements to the dashboard layout and presentation to make the mission intelligence easier to understand and demonstrate.

### 7. Refactor Code

Bob helped organize and refine the project code into reusable components and modules, improving maintainability and readability.

### 8. Test Functionality

Bob assisted with testing the application workflow, including:

* Telemetry processing
* Anomaly detection
* Risk calculation
* Dashboard rendering
* Emergency-event simulation
* AI analysis output

### Development Workflow

```text
Idea
 ↓
IBM Bob AI-Assisted Development
 ↓
Code Generation
 ↓
Implementation
 ↓
Debugging
 ↓
Refactoring
 ↓
Testing
 ↓
Streamlit Prototype
 ↓
GitHub
```

---

# 📈 Results

The current prototype successfully demonstrates an end-to-end simulated mission-intelligence workflow.

### Example Simulated Event

```text
Mission Status:      ALERT
Spacecraft Health:   27%
Mission Risk Score:  73 / 100
Risk Level:          HIGH

Anomaly:
Signal:              battery_voltage
Observed Value:      21.41 V
Detector:            Isolation Forest
```

The detected event is then presented through the dashboard with an AI-assisted explanation and recommended investigation steps.

### Demonstrated Outcome

```text
Raw Telemetry
     ↓
Anomaly Detected
     ↓
Signal Identified
     ↓
Severity Evaluated
     ↓
Risk Score Generated
     ↓
AI Analysis Produced
```

The emergency simulator further demonstrates how multiple simulated faults can move the system from a normal state to an elevated-risk mission state.

---

# 🌍 Real-World Impact

Although AstraGuard AI currently uses simulated data, the concept demonstrates how AI-assisted telemetry intelligence could support future space and satellite operations.

Potential applications include:

* 🛰️ Spacecraft health monitoring
* 📡 Satellite telemetry analysis
* ⚠️ Early anomaly identification
* 📊 Mission-risk visualization
* 🤖 Automated technical reporting
* 🔍 Multi-sensor anomaly correlation
* 🚨 Alert prioritization
* 👩‍🚀 Operator decision support

### Vision

> **Transform complex telemetry streams into clear, explainable mission intelligence.**

---

# 🏆 Challenge Fit

## IBM AI Builders Challenge

**Challenge:** IBM AI Builders Challenge
**Project:** AstraGuard AI
**Theme:** Advance Space Exploration with AI

AstraGuard AI aligns with the challenge by applying artificial intelligence and machine learning to a space-exploration use case.

The project demonstrates:

* AI-assisted development
* Machine-learning-based anomaly detection
* Telemetry intelligence
* Risk assessment
* AI-generated mission analysis
* Interactive visualization
* Emergency simulation

The prototype focuses on demonstrating how AI can help humans interpret complex spacecraft telemetry more efficiently.

---

# 📁 Project Structure

```text
astraguard-ai/
│
├── app.py
│       └── Main Streamlit dashboard
│
├── generate_telemetry.py
│       └── Simulated telemetry generation
│
├── train_anomaly_model.py
│       └── Machine-learning model training
│
├── telemetry.csv
│       └── Simulated telemetry dataset
│
├── requirements.txt
│       └── Python dependencies
│
├── README.md
│       └── Project documentation
│
├── data/
│   ├── telemetry.csv
│   ├── anomaly_results.csv
│   └── risk_results.csv
│
├── models/
│   └── anomaly_model.pkl
│
├── notebooks/
│   └── telemetry_analysis.ipynb
│
└── src/
    ├── ai_explanation.py
    ├── anomaly_detector.py
    └── risk_engine.py
```

---

# ⚙️ Installation

## 1. Clone the Repository

```bash
git clone https://github.com/soujanya04122007/astraguard-ai.git
```

Enter the project directory:

```bash
cd astraguard-ai
```

## 2. Install Dependencies

```bash
pip install -r requirements.txt
```

If required, install the core packages manually:

```bash
pip install streamlit pandas numpy scikit-learn plotly
```

---

# ▶️ Running the Application

Run the Streamlit application:

```bash
python -m streamlit run app.py
```

The application will open in your browser:

```text
http://localhost:8501
```

---

# 🧪 Running the Telemetry Pipeline

To generate simulated telemetry:

```bash
python generate_telemetry.py
```

To train the anomaly-detection model:

```bash
python train_anomaly_model.py
```

Then launch the dashboard:

```bash
python -m streamlit run app.py
```

---

# 📓 Jupyter Notebook

The project includes a telemetry-analysis notebook:

```text
notebooks/telemetry_analysis.ipynb
```

The notebook can be used to demonstrate:

* Dataset exploration
* Telemetry analysis
* Data preparation
* Anomaly analysis
* Visualization
* Model results

---

# 🔮 Future Improvements

Future versions of AstraGuard AI could include:

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

* All telemetry used by this project is simulated.
* This is not an official space-agency system.
* AI-generated analysis is for demonstration purposes.
* Mission Risk Scores are prototype outputs.
* The system must not be used for spacecraft control.
* The system must not be used for safety-critical decisions.

---

# 👩‍💻 Project Information

**Project:** AstraGuard AI
**Domain:** Artificial Intelligence & Space Exploration
**Focus:** Spacecraft Telemetry Anomaly Detection
**Machine Learning:** Isolation Forest
**Dashboard:** Streamlit
**Development:** IBM Bob + VS Code
**Repository:** `https://github.com/soujanya04122007/astraguard-ai`

---

# ⭐ Key Innovation

AstraGuard AI goes beyond simply displaying telemetry.

It connects detection, risk assessment, and explanation into one workflow:

```text
DETECT
  ↓
UNDERSTAND
  ↓
QUANTIFY RISK
  ↓
EXPLAIN
  ↓
INVESTIGATE
```

### 🛰️ AstraGuard AI

**Detect anomalies. Understand risk. Accelerate mission intelligence.**

---

**IBM AI Builders Challenge — August 2026**

> 🛰️ **All telemetry is simulated. Prototype only.**

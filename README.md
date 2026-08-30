# 🛰️ AstraGuard AI

## Space Mission Risk & Anomaly Intelligence

**AstraGuard AI** is an AI-powered prototype for monitoring simulated spacecraft telemetry, detecting anomalous behavior, assessing mission risk, and generating human-readable mission analysis through an interactive Streamlit dashboard.

> ⚠️ **Prototype Disclaimer:** All telemetry used by AstraGuard AI is simulated data created for demonstration and educational purposes. This is not an official NASA, ESA, or space-agency system and must not be used for real spacecraft operations or safety-critical decisions.

---

## 🎯 Problem Statement

Spacecraft continuously generate large volumes of telemetry from critical systems such as:

* Thermal systems
* Battery and power systems
* Communications
* Radiation monitoring
* Fuel
* Solar power generation

Manually monitoring multiple telemetry channels can make it difficult to identify unusual behavior quickly, especially when several subsystem values change simultaneously.

AstraGuard AI addresses this challenge by transforming telemetry data into a unified mission-intelligence workflow:

**Telemetry → Anomaly Detection → Risk Assessment → AI Explanation → Investigation Support**

---

## 💡 Solution

AstraGuard AI combines machine learning, telemetry analytics, risk scoring, visualization, and AI-assisted explanation into a single mission-control-style dashboard.

The system:

1. Loads simulated spacecraft telemetry.
2. Prepares telemetry features for machine learning.
3. Uses an **Isolation Forest** model to identify unusual observations.
4. Calculates a **Mission Risk Index (MRI)**.
5. Identifies the telemetry factors contributing to risk.
6. Generates a human-readable explanation.
7. Provides recommended investigation steps.
8. Displays the results through an interactive Streamlit dashboard.
9. Includes an emergency simulation demonstrating a multi-fault event.

---

# 🚀 Key Features

### 🛰️ 1. Mission Overview

The dashboard provides a centralized view of:

* Mission status
* Spacecraft health
* Mission Risk Score
* Risk level
* Active anomalies
* Latest telemetry timestamp

### 📊 2. Telemetry Monitoring

Interactive Plotly visualizations monitor important telemetry channels, including:

* Temperature
* Battery voltage
* Power consumption
* Radiation level
* Signal strength
* Solar output

Normal operating envelopes are displayed alongside telemetry trends.

### ⚠️ 3. AI Anomaly Detection

A trained **Isolation Forest** model identifies statistically unusual telemetry observations without requiring manually labelled anomaly classes.

The model uses:

* Temperature
* Battery voltage
* Power consumption
* Radiation level
* Signal strength
* Fuel level
* Solar output

### 📈 4. Mission Risk Index

A prototype Mission Risk Index converts telemetry conditions and anomaly information into a risk score from **0–100**.

Risk levels are categorized as:

* **LOW**
* **MEDIUM**
* **HIGH**
* **CRITICAL**

This provides a simpler mission-level interpretation than viewing individual sensor values alone.

### 🤖 5. AI Mission Analysis

AstraGuard AI converts detected events into human-readable explanations containing:

* Detected anomaly
* Telemetry channel
* Observed value
* Severity
* Risk score
* Contributing factors
* Possible interpretation
* Recommended investigation actions

### 🚨 6. Emergency Simulation

The dashboard includes a simulated multi-fault event involving:

* Thermal spike
* Battery voltage drop
* Power surge
* Radiation increase
* Communication signal degradation
* Solar-output reduction

The demonstration follows:

```text
NORMAL
   ↓
ANOMALY
   ↓
HIGH / CRITICAL RISK
   ↓
AI ANALYSIS
   ↓
INVESTIGATION SUPPORT
```

### 💬 7. Ask AstraGuard

The dashboard provides a rule-based mission Q&A interface for questions such as:

* What anomalies were detected?
* Why is the spacecraft at risk?
* What should the operator investigate?
* What is the battery status?
* What is the radiation level?
* What is the spacecraft status?

---

# 🧠 AI / Machine Learning Approach

## Isolation Forest

AstraGuard AI uses the **Isolation Forest** algorithm from `scikit-learn` for unsupervised anomaly detection.

Isolation Forest works by randomly partitioning the feature space. Observations that are easier to isolate from the majority of the data are considered more likely to be anomalous.

### Model Pipeline

```text
Simulated Telemetry
        ↓
Feature Selection
        ↓
StandardScaler
        ↓
Isolation Forest
        ↓
Anomaly Score
        ↓
Anomaly Flag
        ↓
Mission Risk Assessment
        ↓
AI Mission Explanation
```

The trained model is stored as:

```text
models/anomaly_model.pkl
```

The saved model contains the preprocessing and anomaly-detection pipeline used by the dashboard.

---

# 📐 System Architecture

```text
                  ┌─────────────────────────┐
                  │   Simulated Telemetry   │
                  │   data/telemetry.csv    │
                  └────────────┬────────────┘
                               │
                               ▼
                  ┌─────────────────────────┐
                  │   Data Processing       │
                  │   Feature Preparation   │
                  └────────────┬────────────┘
                               │
                               ▼
                  ┌─────────────────────────┐
                  │   StandardScaler        │
                  │   Feature Normalization │
                  └────────────┬────────────┘
                               │
                               ▼
                  ┌─────────────────────────┐
                  │   Isolation Forest      │
                  │   Anomaly Detection     │
                  └────────────┬────────────┘
                               │
                               ▼
                  ┌─────────────────────────┐
                  │   Risk Engine            │
                  │   Mission Risk Index     │
                  └────────────┬────────────┘
                               │
                    ┌──────────┴──────────┐
                    ▼                     ▼
          ┌──────────────────┐   ┌──────────────────┐
          │ Streamlit        │   │ AI Explanation   │
          │ Dashboard        │   │ Engine            │
          └────────┬─────────┘   └────────┬─────────┘
                   │                      │
                   └──────────┬───────────┘
                              ▼
                  ┌─────────────────────────┐
                  │ Mission Intelligence    │
                  │ & Investigation Support │
                  └─────────────────────────┘
```

---

# 🛠️ Technology Stack

| Technology       | Purpose                                 |
| ---------------- | --------------------------------------- |
| Python           | Application and ML development          |
| Streamlit        | Interactive mission-control dashboard   |
| Pandas           | Telemetry data processing               |
| NumPy            | Numerical computation                   |
| scikit-learn     | Isolation Forest and preprocessing      |
| Plotly           | Interactive telemetry visualizations    |
| Joblib           | Model persistence                       |
| Jupyter Notebook | Telemetry analysis and ML demonstration |
| IBM Bob          | AI-assisted development                 |
| Git              | Version control                         |
| GitHub           | Source-code hosting                     |

---

# 📊 Dataset

AstraGuard AI uses a **simulated spacecraft telemetry dataset**.

The main dataset is located at:

```text
data/telemetry.csv
```

The telemetry contains simulated measurements representing spacecraft subsystem behavior, including:

* Temperature
* Battery voltage
* Power consumption
* Radiation level
* Signal strength
* Fuel level
* Solar output

### Generated Results

The project also contains processed outputs:

```text
data/anomaly_results.csv
data/risk_results.csv
```

These files contain the results of anomaly detection and mission-risk processing.

> The dataset is simulated and does not represent official spacecraft telemetry.

---

# 🤖 How IBM Bob Was Used

**IBM Bob was used as the primary AI-assisted development environment during the creation and refinement of AstraGuard AI.**

Bob supported multiple stages of development.

### 1. Generate Telemetry-Processing Code

Bob assisted with developing and refining Python code for:

* Loading telemetry CSV data
* Processing telemetry records
* Selecting ML features
* Preparing data for analysis
* Generating processed outputs

### 2. Develop the ML Pipeline

Bob assisted in structuring the machine-learning workflow from:

```text
Telemetry
    ↓
Preprocessing
    ↓
Feature Preparation
    ↓
Model Training
    ↓
Anomaly Detection
```

### 3. Implement Anomaly Detection

Bob helped implement and refine the `IsolationForest` anomaly-detection pipeline using `scikit-learn`.

### 4. Debug Python and Streamlit Code

Bob was used during development to identify and fix issues involving:

* Python syntax
* Imports
* File paths
* Streamlit execution
* Model loading
* Data processing
* Function integration
* Dashboard errors

### 5. Build Dashboard Components

Bob assisted with the development of dashboard components including:

* Mission Overview
* Telemetry Monitoring
* Active Anomalies
* Mission Risk
* AI Mission Analysis
* Emergency Simulation
* Ask AstraGuard

### 6. Improve User Interface

Bob assisted with improving:

* Dashboard layout
* Dark aerospace theme
* Typography
* Metric cards
* Risk indicators
* Telemetry charts
* Alert components
* Mission-control presentation

### 7. Refactor Code

Bob assisted in organizing the application into reusable modules:

```text
src/
├── anomaly_detector.py
├── risk_engine.py
└── ai_explanation.py
```

This separation improves maintainability and makes the application easier to extend.

### 8. Test Functionality

Bob assisted with testing and debugging the complete workflow:

```text
Telemetry Loading
       ↓
Model Loading
       ↓
Anomaly Detection
       ↓
Risk Assessment
       ↓
AI Explanation
       ↓
Streamlit Dashboard
```

### IBM Bob Development Workflow

```text
Project Idea
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
GitHub Repository
```

---

# 📈 Results

AstraGuard AI demonstrates an end-to-end simulated spacecraft telemetry intelligence pipeline.

The prototype can:

* Process telemetry records
* Detect statistically unusual observations
* Flag anomalous telemetry
* Calculate mission risk
* Identify contributing telemetry factors
* Generate human-readable explanations
* Recommend investigation steps
* Visualize telemetry trends
* Simulate multi-fault emergency conditions

### Example Workflow

```text
Raw Telemetry
     ↓
Data Processing
     ↓
Isolation Forest
     ↓
Anomaly Detected
     ↓
Risk Factors Identified
     ↓
Mission Risk Score
     ↓
AI Mission Analysis
     ↓
Investigation Guidance
```

---

# 🌍 Real-World Impact

Although AstraGuard AI currently uses simulated data, the concept demonstrates how AI-assisted telemetry intelligence could potentially support future space and satellite operations.

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

## IBM AI Builders Challenge — Advance Space Exploration with AI

AstraGuard AI addresses the space-exploration theme by applying artificial intelligence and machine learning to a spacecraft telemetry monitoring scenario.

The project demonstrates:

* AI-assisted development using IBM Bob
* Machine-learning anomaly detection
* Simulated spacecraft telemetry analysis
* Mission risk assessment
* AI-assisted explanations
* Interactive visualization
* Emergency-event simulation
* Investigation support

The goal is to demonstrate how AI can help transform complex telemetry data into understandable mission intelligence.

---

# 📁 Project Structure

```text
astraguard-ai/
│
├── app.py
│   └── Main Streamlit mission-control dashboard
│
├── generate_telemetry.py
│   └── Generates simulated spacecraft telemetry
│
├── train_anomaly_model.py
│   └── Trains and saves the anomaly-detection model
│
├── requirements.txt
│   └── Python dependencies
│
├── README.md
│   └── Project documentation
│
├── assets/
│   └── Dashboard images and project visual assets
│
├── data/
│   ├── telemetry.csv
│   │   └── Simulated spacecraft telemetry dataset
│   │
│   ├── anomaly_results.csv
│   │   └── Anomaly-detection results
│   │
│   └── risk_results.csv
│       └── Mission-risk results
│
├── models/
│   └── anomaly_model.pkl
│       └── Trained anomaly-detection pipeline
│
├── notebooks/
│   └── AstraGuard_Telemetry_Analysis.ipynb
│       └── Telemetry exploration, visualization and ML analysis
│
└── src/
    ├── anomaly_detector.py
    │   └── Telemetry loading, model training and anomaly scoring
    │
    ├── risk_engine.py
    │   └── Mission Risk Index and risk assessment
    │
    └── ai_explanation.py
        └── Human-readable anomaly explanations
```

This structure matches the current repository folders and root files.

---

# ⚙️ Installation

## 1. Clone the Repository

```bash
git clone https://github.com/soujanya04122007/astraguard-ai.git
cd astraguard-ai
```

## 2. Install Dependencies

```bash
pip install -r requirements.txt
```

If necessary, install the core packages manually:

```bash
pip install streamlit pandas numpy scikit-learn plotly joblib
```

---

# ▶️ Running the Application

From the project root directory, run:

```bash
python -m streamlit run app.py
```

The Streamlit dashboard will normally be available at:

```text
http://localhost:8501
```

> **Important:** Use `python -m streamlit run app.py` rather than `python -m py_compile -m streamlit run app.py`. These are different Python commands.

---

# 🧪 Running the ML Pipeline

### Generate simulated telemetry

```bash
python generate_telemetry.py
```

### Train the anomaly-detection model

```bash
python train_anomaly_model.py
```

The trained model is saved to:

```text
models/anomaly_model.pkl
```

### Launch the dashboard

```bash
python -m streamlit run app.py
```

---

# 📓 Telemetry Analysis Notebook

The project includes a Jupyter Notebook:

```text
notebooks/AstraGuard_Telemetry_Analysis.ipynb
```

The notebook demonstrates:

1. Importing libraries
2. Loading simulated telemetry
3. Exploring the dataset
4. Checking missing values
5. Descriptive statistics
6. Visualizing telemetry trends
7. Preparing ML features
8. Training Isolation Forest
9. Identifying anomalous observations
10. Visualizing normal vs anomalous telemetry
11. Calculating anomaly statistics
12. Interpreting the results

---

# 🔮 Future Improvements

Future versions of AstraGuard AI could include:

* Real-time telemetry streaming
* Advanced time-series anomaly detection
* LSTM or Transformer-based forecasting
* Automated anomaly classification
* Multi-sensor anomaly correlation
* More advanced explainable AI
* Historical mission comparison
* Real satellite telemetry integration
* Automated alert notifications
* Digital-twin simulation
* Fault-propagation modeling
* Mission-control role-based interfaces

---

# 🔐 Safety & Data Disclaimer

AstraGuard AI is a **research and demonstration prototype**.

* All telemetry is simulated.
* The system is not an official NASA, ESA, or space-agency system.
* AI-generated analysis is for demonstration purposes.
* Mission Risk Index values are prototype outputs.
* The application must not be used to control spacecraft.
* The application must not be used for safety-critical decisions.

---

# ⭐ Key Innovation

AstraGuard AI goes beyond simply displaying telemetry.

It connects **detection, understanding, risk quantification, explanation, and investigation support** into a single workflow:

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

## 📌 Project Information

| Item             | Details                                     |
| ---------------- | ------------------------------------------- |
| Project          | AstraGuard AI                               |
| Domain           | Artificial Intelligence & Space Exploration |
| Focus            | Spacecraft Telemetry Anomaly Detection      |
| Machine Learning | Isolation Forest                            |
| Dashboard        | Streamlit                                   |
| Development      | IBM Bob + VS Code                           |
| Repository       | GitHub                                      |
| Data             | Simulated telemetry                         |
| Challenge Theme  | Advance Space Exploration with AI           |

---

**IBM AI Builders Challenge — August 2026**

> 🛰️ **All telemetry is simulated. Prototype only.**

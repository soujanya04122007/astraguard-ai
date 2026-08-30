\# 🛰️ AstraGuard AI

## Spacecraft Telemetry Anomaly Detection & Mission Risk Intelligence

**IBM AI Builders Challenge — August 2026**
**Theme: Advance Space Exploration with AI**

> ⚠️ **DISCLAIMER:** AstraGuard AI is an AI prototype built using **simulated spacecraft telemetry**. It is not real NASA, ESA, or space-agency telemetry and must not be used for real mission operations.

---

## 📌 Problem Statement

Space missions continuously generate large amounts of telemetry data from spacecraft systems such as temperature, battery voltage, power consumption, radiation, communication signal strength, fuel level, and solar power generation.

Detecting abnormal patterns early is important because a combination of unusual readings may indicate a potential spacecraft-system problem.

Traditional threshold-based monitoring can identify values outside predefined limits, but it may not effectively identify unusual combinations or patterns across multiple telemetry channels.

**AstraGuard AI** addresses this challenge by combining machine-learning-based anomaly detection with a mission risk scoring and AI explanation layer.

---

## 💡 Solution

AstraGuard AI is a spacecraft telemetry intelligence prototype that analyzes simulated telemetry and transforms raw sensor readings into actionable mission insights.

The system:

1. Loads spacecraft telemetry data.
2. Processes multiple telemetry channels.
3. Uses **IsolationForest** for unsupervised anomaly detection.
4. Identifies unusual telemetry observations.
5. Calculates a **Mission Risk Index (MRI)**.
6. Classifies mission risk into:

   * 🟢 LOW
   * 🟡 MEDIUM
   * 🔴 HIGH
   * 🟣 CRITICAL
7. Identifies contributing telemetry factors.
8. Generates an AI-style explanation of detected events.
9. Provides recommended investigation steps.
10. Displays the results through an interactive **Streamlit Mission Control Dashboard**.
11. Includes an emergency simulation demonstrating a multi-fault event.

---

# 🚀 Key Features

### 1. Telemetry Monitoring

AstraGuard AI monitors important spacecraft telemetry channels:

* Temperature
* Battery voltage
* Power consumption
* Radiation level
* Signal strength
* Fuel level
* Solar output

---

### 2. Machine Learning Anomaly Detection

The system uses the **IsolationForest** algorithm from scikit-learn.

IsolationForest is an unsupervised anomaly-detection algorithm that identifies observations that are easier to isolate from the rest of the dataset.

The model analyzes multiple telemetry features simultaneously instead of relying only on individual threshold checks.

---

### 3. Data Preprocessing

The anomaly-detection pipeline uses:

```text
Telemetry Data
      ↓
Feature Selection
      ↓
StandardScaler
      ↓
IsolationForest
      ↓
Anomaly Score
      ↓
Anomaly / Normal Classification
```

The model features include:

```text
temperature
battery_voltage
power_consumption
radiation_level
signal_strength
fuel_level
solar_output
```

---

### 4. Mission Risk Index

AstraGuard AI includes a prototype **Mission Risk Index (MRI)** that converts telemetry conditions and anomaly information into a risk score from:

```text
0 ─────────────────────── 100
Low                         Critical
```

The system classifies risk into four levels:

|  Score | Risk Level |
| -----: | ---------- |
|   0–29 | LOW        |
|  30–60 | MEDIUM     |
|  61–80 | HIGH       |
| 81–100 | CRITICAL   |

The MRI is a prototype decision-support mechanism for demonstration purposes.

---

### 5. AI Mission Analysis

When an anomaly is detected, AstraGuard AI provides an explanation describing:

* What happened
* Which telemetry channels contributed
* How severe the deviation is
* Why the event may be important
* What the operator should investigate

The explanation layer is designed for **prototype decision support**, not autonomous spacecraft control.

---

### 6. Emergency Simulation

The dashboard includes a simulated emergency event that injects multiple abnormal telemetry conditions simultaneously.

Example simulated event:

```text
Temperature       → 112.4 °C
Battery Voltage   → 19.8 V
Power Consumption → 331 W
Radiation         → 11.2 mSv/h
Signal Strength   → -121 dBm
Solar Output      → 8.7 W
```

The dashboard demonstrates the complete workflow:

```text
NORMAL
   ↓
ANOMALY DETECTION
   ↓
HIGH / CRITICAL RISK
   ↓
AI MISSION ANALYSIS
   ↓
INVESTIGATION RECOMMENDATIONS
```

---

### 7. Ask AstraGuard

The dashboard includes a telemetry Q&A interface.

Example questions:

```text
What anomalies were detected?

Why is the spacecraft at risk?

What should the operator investigate?

What is the battery status?

What is the radiation level?
```

The Q&A system uses the processed telemetry and risk results to provide contextual responses.

---

# 🤖 AI Approach

AstraGuard AI uses an unsupervised machine-learning approach.

## IsolationForest

The core anomaly detector is:

```python
IsolationForest
```

Configuration:

```text
n_estimators = 200
contamination = 0.05
random_state = 42
n_jobs = -1
```

The model is trained without requiring manually labelled anomaly classes.

### Feature Pipeline

```text
Simulated Telemetry
        ↓
Select 7 telemetry features
        ↓
StandardScaler
        ↓
IsolationForest
        ↓
Decision Function
        ↓
Anomaly Score
        ↓
Normal / Anomaly
```

The system uses the IsolationForest prediction convention:

```text
+1 → Normal
-1 → Anomaly
```

---

# 🏗️ System Architecture

```text
                 ┌──────────────────────────┐
                 │ Simulated Telemetry CSV  │
                 │      telemetry.csv       │
                 └────────────┬─────────────┘
                              │
                              ▼
                 ┌──────────────────────────┐
                 │   Telemetry Processing   │
                 │      Pandas / NumPy      │
                 └────────────┬─────────────┘
                              │
                              ▼
                 ┌──────────────────────────┐
                 │    StandardScaler        │
                 │   Feature Normalization  │
                 └────────────┬─────────────┘
                              │
                              ▼
                 ┌──────────────────────────┐
                 │    IsolationForest       │
                 │   Anomaly Detection      │
                 └────────────┬─────────────┘
                              │
                              ▼
                 ┌──────────────────────────┐
                 │     Risk Engine           │
                 │    Mission Risk Index     │
                 └────────────┬─────────────┘
                              │
                              ▼
                 ┌──────────────────────────┐
                 │    AI Explanation Layer  │
                 │ Analysis & Recommendations│
                 └────────────┬─────────────┘
                              │
                              ▼
                 ┌──────────────────────────┐
                 │ Streamlit Mission Control│
                 │        Dashboard         │
                 └──────────────────────────┘
```

---

# 📁 Repository Structure

```text
astraguard-ai/
│
├── app.py
├── README.md
├── requirements.txt
│
├── data/
│   ├── telemetry.csv
│   └── anomaly_results.csv
│
├── models/
│   └── anomaly_model.pkl
│
├── notebooks/
│   └── telemetry_analysis.ipynb
│
├── src/
│   ├── anomaly_detector.py
│   ├── risk_engine.py
│   └── ai_explanation.py
│
└── dashboard/
    └── dashboard screenshots / assets
```

> Keep the repository structure consistent with the actual files in your GitHub repository. If your screenshot/asset files have different names, keep their existing names rather than creating duplicate files.

---

# 🛠️ Technology Stack

| Technology       | Purpose                                |
| ---------------- | -------------------------------------- |
| Python           | Core programming language              |
| Pandas           | Telemetry data processing              |
| NumPy            | Numerical computation                  |
| Scikit-learn     | Machine learning                       |
| IsolationForest  | Anomaly detection                      |
| StandardScaler   | Feature normalization                  |
| Joblib           | ML model persistence                   |
| Streamlit        | Interactive dashboard                  |
| Plotly           | Telemetry visualizations               |
| Jupyter Notebook | ML experimentation and analysis        |
| GitHub           | Version control and project repository |
| IBM Bob          | AI-assisted development                |

---

# 📊 Dataset

AstraGuard AI uses **simulated spacecraft telemetry**.

The dataset contains telemetry measurements representing spacecraft-system behavior over time.

Example fields:

```text
timestamp
temperature
battery_voltage
power_consumption
radiation_level
signal_strength
fuel_level
solar_output
```

The telemetry is intentionally simulated for the prototype.

### Important

The dataset does **not** represent real spacecraft telemetry.

It should not be interpreted as NASA, ESA, ISRO, or any other agency's operational data.

---

# 🧪 Telemetry Analysis Notebook

The project includes a Jupyter notebook:

```text
notebooks/telemetry_analysis.ipynb
```

The notebook demonstrates the machine-learning workflow:

1. Import libraries
2. Load simulated telemetry
3. Explore the dataset
4. Check missing values
5. Generate descriptive statistics
6. Visualize telemetry trends
7. Prepare ML features
8. Train IsolationForest
9. Detect anomalies
10. Visualize normal vs anomalous observations
11. Calculate anomaly statistics
12. Interpret results

The notebook provides a transparent view of how the anomaly-detection model was developed.

---

# 🤖 How IBM Bob Was Used

**IBM Bob was used as the primary AI-assisted development environment throughout the project.**

Bob supported the development workflow in several areas:

### 1. Generate Telemetry-Processing Code

IBM Bob helped generate and structure Python code for:

* Loading telemetry CSV files
* Parsing timestamps
* Selecting telemetry features
* Processing Pandas DataFrames
* Preparing data for machine learning

---

### 2. Develop ML Pipeline

Bob helped develop the machine-learning pipeline consisting of:

```text
Telemetry
   ↓
Feature Selection
   ↓
StandardScaler
   ↓
IsolationForest
   ↓
Anomaly Scores
   ↓
Anomaly Labels
```

It also helped structure the reusable model-training functions.

---

### 3. Implement Anomaly Detection

IBM Bob assisted with implementing the IsolationForest anomaly detector.

The implementation includes:

* Model configuration
* Training
* Anomaly scoring
* Prediction
* Result labelling
* Model persistence using Joblib

The trained model is saved as:

```text
models/anomaly_model.pkl
```

---

### 4. Debug Python / Streamlit Code

Bob was used to identify and resolve development issues involving:

* Python syntax errors
* Import paths
* Streamlit execution
* Model loading
* Data-processing errors
* Dashboard runtime issues
* File and folder organization

This helped turn individual Python modules into an integrated application.

---

### 5. Build Dashboard Components

Bob helped develop the Streamlit Mission Control dashboard, including:

* Mission overview
* Mission status
* Spacecraft health
* Mission Risk Index
* Telemetry charts
* Active anomaly table
* AI Mission Analysis
* Emergency simulation
* Ask AstraGuard Q&A

---

### 6. Improve UI

IBM Bob assisted with improving the dashboard's visual presentation.

The dashboard uses a dark aerospace-inspired interface with:

* Mission-control styling
* Risk badges
* KPI cards
* Telemetry charts
* Risk gauge
* Alert panels
* Emergency-event visualization
* Responsive Streamlit columns

The goal was to make the technical results understandable to both technical and non-technical reviewers.

---

### 7. Refactor Code

Bob helped organize the project into reusable modules rather than placing all functionality inside a single Python file.

The main modules include:

```text
src/anomaly_detector.py
src/risk_engine.py
src/ai_explanation.py
app.py
```

This separation improves readability, maintainability, and reuse.

---

### 8. Test Functionality

IBM Bob was also used to help test the integrated prototype.

Testing covered:

* Telemetry loading
* Model loading
* Anomaly scoring
* Risk calculation
* Dashboard rendering
* Emergency simulation
* AI explanation generation
* Q&A functionality

The final prototype connects these components into one workflow.

---

# 📈 Results

AstraGuard AI successfully demonstrates an end-to-end telemetry intelligence pipeline:

```text
Telemetry Data
      ↓
Machine Learning
      ↓
Anomaly Detection
      ↓
Risk Scoring
      ↓
AI Explanation
      ↓
Operator-Oriented Dashboard
```

The system can:

* Detect statistically unusual telemetry observations.
* Assign anomaly scores.
* Identify contributing telemetry factors.
* Calculate a prototype Mission Risk Index.
* Categorize risk severity.
* Generate investigation recommendations.
* Visualize telemetry and anomalies.
* Simulate a multi-fault spacecraft event.

Because the dataset is simulated, the results demonstrate **technical feasibility rather than real-world spacecraft performance**.

---

# 🌍 Real-World Impact

A system like AstraGuard AI could potentially support future spacecraft monitoring by helping mission teams:

* Detect unusual telemetry patterns earlier.
* Reduce the amount of raw telemetry that operators must manually inspect.
* Highlight potentially important multi-channel events.
* Prioritize telemetry events based on risk.
* Provide understandable explanations for detected anomalies.
* Support faster investigation and decision-making.

The prototype demonstrates how AI could complement human mission operators rather than replace them.

---

# 🏆 Challenge Fit

AstraGuard AI directly addresses the challenge theme:

## 🚀 Advance Space Exploration with AI

The project demonstrates the use of artificial intelligence for spacecraft telemetry intelligence.

It combines:

```text
AI / Machine Learning
        +
Space Exploration
        +
Anomaly Detection
        +
Risk Intelligence
        +
Human-Centered Decision Support
```

The prototype shows how AI can transform simulated spacecraft telemetry into interpretable mission insights.

---

# 💻 Installation

## 1. Clone the repository

```bash
git clone https://github.com/YOUR-USERNAME/astraguard-ai.git
```

```bash
cd astraguard-ai
```

Replace `YOUR-USERNAME` with your GitHub username.

---

## 2. Create a virtual environment

Windows:

```bash
python -m venv .venv
```

Activate it:

```bash
.venv\Scripts\activate
```

---

## 3. Install dependencies

```bash
python -m pip install -r requirements.txt
```

If Streamlit is not recognized, install it with:

```bash
python -m pip install streamlit
```

You can then verify it with:

```bash
python -m streamlit --version
```

---

# ▶️ Running the Application

From the project root directory:

```bash
python -m streamlit run app.py
```

### Important

Use:

```bash
python -m streamlit run app.py
```

instead of:

```bash
python.exe -m py_compile -m streamlit run app.py
```

`py_compile` is only for checking Python syntax. It is **not** the command used to start Streamlit.

---

# 🧠 Model

The trained anomaly-detection pipeline is stored at:

```text
models/anomaly_model.pkl
```

The model contains the preprocessing and anomaly-detection pipeline.

The application loads the saved pipeline rather than retraining the model every time the dashboard starts.

---

# 📄 Main Source Files

### `app.py`

Main Streamlit application.

Responsible for:

* Dashboard UI
* Telemetry visualization
* Mission overview
* Emergency simulation
* AI analysis
* Q&A interface

### `src/anomaly_detector.py`

Responsible for:

* Telemetry loading
* Feature selection
* IsolationForest pipeline
* Model training
* Anomaly scoring
* Model saving/loading
* Detection statistics

### `src/risk_engine.py`

Responsible for:

* Telemetry risk assessment
* Mission Risk Index
* Risk levels
* Risk contributions
* Batch risk summaries

### `src/ai_explanation.py`

Responsible for:

* AI-style mission explanations
* Contributing-factor interpretation
* Investigation recommendations

### `notebooks/telemetry_analysis.ipynb`

Provides the exploratory machine-learning analysis and visualization workflow.

---

# 🔮 Future Improvements

Future versions of AstraGuard AI could include:

* Real-time telemetry streaming
* Online anomaly detection
* More advanced time-series models
* LSTM/Transformer-based sequence detection
* Multivariate temporal anomaly detection
* Explainable AI techniques such as SHAP
* Real spacecraft telemetry datasets where legally and appropriately available
* Automated alert prioritization
* Historical mission comparison
* Role-based mission dashboards
* Cloud deployment
* Secure telemetry ingestion
* Human-in-the-loop operator feedback
* More advanced AI-powered mission assistant capabilities

---

# ⚠️ Prototype Safety Notice

AstraGuard AI is a **demonstration and educational prototype**.

All telemetry data is simulated.

The anomaly detector, Mission Risk Index, emergency simulation, and AI explanations are designed to demonstrate the concept of AI-assisted spacecraft monitoring.

**This system is not an official NASA, ESA, ISRO, or other space-agency system and must not be used for real spacecraft operations or safety-critical decisions.**

---

# 👩‍💻 Project

**AstraGuard AI**

### IBM AI Builders Challenge — August 2026

**Theme:** Advance Space Exploration with AI

Built as an AI-powered prototype for spacecraft telemetry anomaly detection, mission risk intelligence, and human-centered decision support.

---

## ⭐ Project Pipeline

```text
SIMULATED TELEMETRY
        │
        ▼
DATA PROCESSING
        │
        ▼
FEATURE NORMALIZATION
        │
        ▼
ISOLATION FOREST
        │
        ▼
ANOMALY DETECTION
        │
        ▼
MISSION RISK INDEX
        │
        ▼
AI EXPLANATION
        │
        ▼
MISSION CONTROL DASHBOARD
        │
        ▼
OPERATOR INSIGHTS
```

**AstraGuard AI — Turning spacecraft telemetry into mission intelligence.** 🛰️

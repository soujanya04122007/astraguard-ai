# 🛰️ AstraGuard AI

## Spacecraft Telemetry Anomaly Detection & Mission Risk Intelligence

**IBM AI Builders Challenge — August 2026**
**Challenge Theme: Advance Space Exploration with AI**

> ⚠️ **DISCLAIMER:** AstraGuard AI is an AI prototype built using **simulated spacecraft telemetry**. It is not real NASA, ESA, ISRO, or other space-agency telemetry and must not be used for real mission operations or safety-critical decisions.

---

## 📌 Problem Statement

Spacecraft continuously generate telemetry from multiple systems, including temperature, battery voltage, power consumption, radiation, communication signal strength, fuel level, and solar power generation.

Monitoring these channels manually can make it difficult to identify unusual patterns quickly, especially when several telemetry values change simultaneously.

Traditional threshold-based monitoring can identify individual values outside predefined limits, but it may not effectively identify unusual combinations of multiple telemetry variables.

**AstraGuard AI** addresses this challenge by combining machine-learning-based anomaly detection with a prototype Mission Risk Index and AI-assisted explanations.

---

## 💡 Solution

AstraGuard AI is a spacecraft telemetry intelligence prototype that converts simulated telemetry into interpretable mission insights.

The system:

1. Loads simulated spacecraft telemetry.
2. Processes multiple telemetry channels.
3. Normalizes machine-learning features.
4. Uses **IsolationForest** for unsupervised anomaly detection.
5. Generates anomaly scores and anomaly labels.
6. Evaluates telemetry conditions using a prototype risk engine.
7. Calculates a **Mission Risk Index (MRI)**.
8. Classifies risk into LOW, MEDIUM, HIGH, or CRITICAL.
9. Identifies contributing telemetry factors.
10. Generates AI-style mission explanations.
11. Provides investigation recommendations.
12. Displays the results through an interactive Streamlit Mission Control dashboard.
13. Provides an emergency simulation demonstrating a simultaneous multi-fault event.

---

# 🚀 Key Features

## 1. Telemetry Monitoring

AstraGuard AI monitors the following simulated spacecraft telemetry channels:

* 🌡️ Temperature
* 🔋 Battery voltage
* ⚡ Power consumption
* ☢️ Radiation level
* 📡 Signal strength
* ⛽ Fuel level
* ☀️ Solar output

---

## 2. Machine Learning Anomaly Detection

The core anomaly detector uses **IsolationForest** from scikit-learn.

IsolationForest is an unsupervised anomaly-detection algorithm that identifies observations that can be isolated more easily from the rest of the dataset.

This allows AstraGuard AI to detect unusual telemetry patterns without requiring manually labelled anomaly classes.

### Model Configuration

```text
Algorithm: IsolationForest
n_estimators: 200
contamination: 0.05
random_state: 42
n_jobs: -1
```

The model uses these telemetry features:

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

## 3. Machine Learning Pipeline

The anomaly-detection pipeline is:

```text
Simulated Telemetry
        ↓
Feature Selection
        ↓
StandardScaler
        ↓
IsolationForest
        ↓
Anomaly Score
        ↓
Anomaly Classification
```

The IsolationForest prediction convention is:

```text
+1 → Normal
-1 → Anomaly
```

The trained pipeline is saved as:

```text
models/anomaly_model.pkl
```

---

# 📊 Mission Risk Intelligence

AstraGuard AI includes a prototype **Mission Risk Index (MRI)** that converts telemetry conditions and anomaly information into a risk score between 0 and 100.

```text
0 ─────────────────────────────── 100
LOW                              CRITICAL
```

The prototype risk levels are:

| Risk Score | Risk Level |
| ---------: | ---------- |
|       0–29 | LOW        |
|      30–60 | MEDIUM     |
|      61–80 | HIGH       |
|     81–100 | CRITICAL   |

The MRI is intended as a **demonstration decision-support mechanism**, not as a real spacecraft safety standard.

---

# 🤖 AI Mission Analysis

When an anomalous event is identified, AstraGuard AI provides an interpretable analysis.

The explanation layer can identify:

* The detected event
* Contributing telemetry channels
* Severity of telemetry deviations
* Potential reasons for concern
* Recommended investigation steps

The explanations are intended to help a human operator understand the model output.

> The AI explanation layer is a prototype and does not provide autonomous spacecraft control.

---

# ⚡ Emergency Simulation

The dashboard contains an interactive emergency simulation.

When activated, the prototype injects a simulated simultaneous multi-fault event.

Example:

```text
Temperature       → 112.4 °C
Battery Voltage   → 19.8 V
Power Consumption → 331.0 W
Radiation Level   → 11.2 mSv/h
Signal Strength   → -121 dBm
Solar Output      → 8.7 W
```

The dashboard demonstrates the complete response pipeline:

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

This is a **simulated demonstration event**, not a real spacecraft emergency.

---

# 💬 Ask AstraGuard

The dashboard includes an interactive Q&A section.

Example questions include:

```text
What anomalies were detected?

Why is the spacecraft at risk?

What should the operator investigate?

What is the battery status?

What is the radiation level?
```

The Q&A engine uses the processed telemetry, anomaly results, and risk information to provide contextual responses.

---

# 🏗️ System Architecture

```text
                 ┌─────────────────────────┐
                 │ Simulated Telemetry CSV │
                 │     telemetry.csv       │
                 └────────────┬────────────┘
                              │
                              ▼
                 ┌─────────────────────────┐
                 │ Telemetry Processing    │
                 │     Pandas / NumPy      │
                 └────────────┬────────────┘
                              │
                              ▼
                 ┌─────────────────────────┐
                 │     StandardScaler      │
                 │  Feature Normalization  │
                 └────────────┬────────────┘
                              │
                              ▼
                 ┌─────────────────────────┐
                 │     IsolationForest     │
                 │   Anomaly Detection     │
                 └────────────┬────────────┘
                              │
                              ▼
                 ┌─────────────────────────┐
                 │      Risk Engine        │
                 │   Mission Risk Index    │
                 └────────────┬────────────┘
                              │
                              ▼
                 ┌─────────────────────────┐
                 │   AI Explanation Layer  │
                 │ Analysis & Recommendations│
                 └────────────┬────────────┘
                              │
                              ▼
                 ┌─────────────────────────┐
                 │ Streamlit Mission Control│
                 │        Dashboard        │
                 └─────────────────────────┘
```

---

# 📁 Repository Structure

```text
astraguard-ai/
│
├── assets/
│   └── Dashboard screenshots and project visuals
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
├── src/
│   ├── anomaly_detector.py
│   ├── risk_engine.py
│   └── ai_explanation.py
│
├── app.py
├── generate_telemetry.py
├── train_anomaly_model.py
├── requirements.txt
└── README.md
```

---

# 📂 File Descriptions

| File / Folder                        | Purpose                                                                             |
| ------------------------------------ | ----------------------------------------------------------------------------------- |
| `app.py`                             | Main Streamlit Mission Control dashboard                                            |
| `generate_telemetry.py`              | Generates simulated spacecraft telemetry                                            |
| `train_anomaly_model.py`             | Trains and saves the anomaly-detection model                                        |
| `requirements.txt`                   | Python dependencies                                                                 |
| `README.md`                          | Project documentation                                                               |
| `data/telemetry.csv`                 | Simulated spacecraft telemetry dataset                                              |
| `data/anomaly_results.csv`           | Telemetry data with anomaly-detection results                                       |
| `data/risk_results.csv`              | Telemetry data with prototype risk results                                          |
| `models/anomaly_model.pkl`           | Saved trained ML pipeline                                                           |
| `src/anomaly_detector.py`            | Telemetry loading, preprocessing, anomaly detection, scoring, and model persistence |
| `src/risk_engine.py`                 | Mission Risk Index and telemetry risk assessment                                    |
| `src/ai_explanation.py`              | AI-style explanations and investigation recommendations                             |
| `notebooks/telemetry_analysis.ipynb` | Exploratory telemetry analysis and ML workflow                                      |
| `assets/`                            | Dashboard screenshots and project visuals                                           |

---

# 🔄 Project Workflow

```text
generate_telemetry.py
        │
        ▼
data/telemetry.csv
        │
        ▼
train_anomaly_model.py
        │
        ▼
models/anomaly_model.pkl
        │
        ▼
app.py
        │
        ├── anomaly_detector.py
        │          ↓
        │    Anomaly Detection
        │
        ├── risk_engine.py
        │          ↓
        │    Mission Risk Index
        │
        └── ai_explanation.py
                   ↓
             AI Mission Analysis
                   │
                   ▼
          Streamlit Dashboard
```

---

# 🧪 Telemetry Analysis Notebook

The project includes:

```text
notebooks/telemetry_analysis.ipynb
```

The notebook demonstrates the machine-learning workflow used during development.

### Notebook Steps

1. Import libraries
2. Load simulated telemetry
3. Explore the dataset
4. Check missing values
5. Generate descriptive statistics
6. Visualize telemetry trends
7. Prepare machine-learning features
8. Train IsolationForest
9. Identify anomalous observations
10. Visualize normal versus anomalous telemetry
11. Calculate anomaly statistics
12. Interpret the results

The notebook provides a transparent view of the data-analysis and anomaly-detection process.

---

# 🛠️ Technology Stack

| Technology       | Purpose                                |
| ---------------- | -------------------------------------- |
| Python           | Core programming language              |
| Pandas           | Data processing                        |
| NumPy            | Numerical computation                  |
| Scikit-learn     | Machine learning                       |
| IsolationForest  | Unsupervised anomaly detection         |
| StandardScaler   | Feature normalization                  |
| Joblib           | Model persistence                      |
| Streamlit        | Interactive dashboard                  |
| Plotly           | Telemetry visualization                |
| Jupyter Notebook | Data analysis and experimentation      |
| GitHub           | Version control and project repository |
| IBM Bob          | AI-assisted development                |

---

# 🤖 How IBM Bob Was Used

**IBM Bob was used as the primary AI-assisted development environment for building and refining AstraGuard AI.**

Bob supported the project development process in the following areas:

## 1. Generate Telemetry-Processing Code

IBM Bob helped generate and structure Python code for:

* Loading telemetry CSV files
* Parsing timestamps
* Selecting telemetry features
* Processing Pandas DataFrames
* Preparing data for machine learning

---

## 2. Develop ML Pipeline

Bob assisted in developing the machine-learning pipeline:

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
Normal / Anomaly Classification
```

It also helped structure reusable training and inference functions.

---

## 3. Implement Anomaly Detection

IBM Bob assisted with implementing the IsolationForest anomaly detector, including:

* Model configuration
* Model training
* Anomaly scoring
* Prediction
* Result labelling
* Model persistence using Joblib

The trained model is stored at:

```text
models/anomaly_model.pkl
```

---

## 4. Debug Python / Streamlit Code

Bob helped identify and resolve development issues involving:

* Python syntax errors
* Import paths
* Streamlit execution
* Model loading
* Data-processing issues
* Dashboard runtime issues
* Project file organization

---

## 5. Build Dashboard Components

Bob helped develop the Streamlit Mission Control dashboard components, including:

* Mission Overview
* Spacecraft Health
* Mission Risk Score
* Risk Level
* Telemetry Monitoring
* Active Anomalies
* Mission Risk Gauge
* AI Mission Analysis
* Emergency Simulation
* Ask AstraGuard Q&A

---

## 6. Improve UI

IBM Bob assisted with improving the dashboard interface and presentation.

The dashboard uses a dark aerospace-inspired mission-control design with:

* KPI cards
* Telemetry charts
* Risk badges
* Risk gauge
* Alert panels
* Anomaly tables
* Emergency-event visualization
* Responsive dashboard columns

The objective was to make machine-learning results easier to understand and interpret.

---

## 7. Refactor Code

Bob helped organize functionality into reusable modules instead of keeping the entire application in one Python file.

The project separates major responsibilities into:

```text
src/anomaly_detector.py
src/risk_engine.py
src/ai_explanation.py
app.py
```

This improves readability, maintainability, and modularity.

---

## 8. Test Functionality

IBM Bob assisted with testing the integrated prototype, including:

* Telemetry loading
* Model loading
* Anomaly scoring
* Risk calculation
* Dashboard rendering
* Emergency simulation
* AI explanation generation
* Q&A functionality

---

# 📊 Dataset

AstraGuard AI uses **simulated spacecraft telemetry**.

The dataset contains time-based measurements representing spacecraft-system behavior.

### Main telemetry fields

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

The data is generated specifically for this AI prototype.

> **The dataset does not represent real NASA, ESA, ISRO, or other space-agency telemetry.**

---

# 📈 Results

AstraGuard AI demonstrates an end-to-end telemetry intelligence workflow:

```text
Telemetry
    ↓
Machine Learning
    ↓
Anomaly Detection
    ↓
Risk Assessment
    ↓
AI Explanation
    ↓
Mission Control Dashboard
```

The prototype can:

* Detect statistically unusual telemetry observations.
* Generate anomaly scores.
* Classify observations as normal or anomalous.
* Identify contributing telemetry factors.
* Calculate a prototype Mission Risk Index.
* Categorize risk severity.
* Generate investigation recommendations.
* Visualize telemetry trends.
* Display anomaly events.
* Simulate a multi-fault event.
* Provide contextual telemetry Q&A.

Because the dataset is simulated, these results demonstrate **prototype functionality and technical feasibility**, not real spacecraft performance.

---

# 🌍 Real-World Impact

A future system based on this concept could potentially assist mission teams by:

* Detecting unusual telemetry patterns earlier.
* Reducing the amount of raw telemetry requiring manual inspection.
* Highlighting potentially important multi-channel events.
* Prioritizing events according to estimated risk.
* Providing interpretable explanations of anomalies.
* Supporting faster human investigation and decision-making.

The intended concept is **human-centered AI decision support**, where AI assists operators rather than autonomously controlling spacecraft.

---

# 🏆 Challenge Fit

## 🚀 Advance Space Exploration with AI

AstraGuard AI applies artificial intelligence and machine learning to a space-exploration monitoring scenario.

The project combines:

```text
Artificial Intelligence
        +
Machine Learning
        +
Space Exploration
        +
Telemetry Analysis
        +
Anomaly Detection
        +
Mission Risk Intelligence
        +
Human-Centered Decision Support
```

The prototype demonstrates how AI can transform simulated spacecraft telemetry into interpretable mission insights.

---

# 💻 Installation

## 1. Clone the repository

```bash
git clone https://github.com/soujanya04122007/astraguard-ai.git
```

Enter the project directory:

```bash
cd astraguard-ai
```

---

## 2. Create a virtual environment

On Windows:

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

If Streamlit is not installed:

```bash
python -m pip install streamlit
```

Verify the installation:

```bash
python -m streamlit --version
```

---

# ▶️ Running the Application

From the project root directory, run:

```bash
python -m streamlit run app.py
```

### Important

Use:

```bash
python -m streamlit run app.py
```

Do **not** use:

```bash
python.exe -m py_compile -m streamlit run app.py
```

`py_compile` is used for Python syntax checking and is not the command for starting a Streamlit application.

---

# 🧠 Trained Model

The trained anomaly-detection pipeline is stored at:

```text
models/anomaly_model.pkl
```

The saved pipeline contains the preprocessing and IsolationForest components required for anomaly scoring.

The Streamlit application loads the saved model rather than retraining the model every time the dashboard starts.

---

# 🔐 Prototype Safety Notice

AstraGuard AI is a **demonstration and educational prototype**.

All telemetry data used by this project is simulated.

The anomaly detector, Mission Risk Index, emergency simulation, and AI explanations are designed to demonstrate the concept of AI-assisted spacecraft monitoring.

**This project is not an official NASA, ESA, ISRO, or other space-agency system and must not be used for real spacecraft operations or safety-critical decisions.**

---

# 🔮 Future Improvements

Future versions could include:

* Real-time telemetry streaming
* Online anomaly detection
* Advanced time-series anomaly detection
* LSTM or Transformer-based models
* Multivariate temporal analysis
* Explainable AI techniques
* Human-in-the-loop operator feedback
* Automated alert prioritization
* Historical mission comparison
* Cloud deployment
* Secure telemetry ingestion
* More advanced AI mission-assistant capabilities

---

# 👩‍💻 Project Information

**Project:** AstraGuard AI
**Challenge:** IBM AI Builders Challenge — August 2026
**Theme:** Advance Space Exploration with AI

### Core Technologies

**Python · Scikit-learn · IsolationForest · Pandas · NumPy · Streamlit · Plotly · Jupyter · IBM Bob**

---

## 🛰️ AstraGuard AI

> **Turning spacecraft telemetry into mission intelligence.**

**Simulated data. AI-powered prototype. Human-centered decision support.**

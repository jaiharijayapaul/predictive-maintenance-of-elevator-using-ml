# 🛗 Elevator Predictive Maintenance System

> **AI-powered predictive maintenance for elevators** — detects failures before they occur using machine learning analysis of real-time sensor data.

[![Python](https://img.shields.io/badge/Python-3.11-blue)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.35-red)](https://streamlit.io)
[![Scikit-learn](https://img.shields.io/badge/scikit--learn-1.5-orange)](https://scikit-learn.org)
[![SHAP](https://img.shields.io/badge/SHAP-0.45-purple)](https://shap.readthedocs.io)

---

## 📋 Table of Contents

- [Project Overview](#project-overview)
- [Problem Statement](#problem-statement)
- [System Architecture](#system-architecture)
- [Dataset](#dataset)
- [Feature Engineering](#feature-engineering)
- [Machine Learning Models](#machine-learning-models)
- [Model Results](#model-results)
- [Application Pages](#application-pages)
- [Installation](#installation)
- [Usage](#usage)
- [Deployment](#deployment)
- [Project Structure](#project-structure)
- [Technologies Used](#technologies-used)

---

## 🎯 Project Overview

This system is a **complete AI + Machine Learning predictive maintenance solution** for elevators that:

- Analyzes **16+ sensor parameters** in real-time
- Predicts elevator health as **Healthy / Maintenance Required / Failure Predicted**
- Generates **intelligent maintenance recommendations** with priority ordering
- Provides **SHAP-based AI explanations** for every prediction
- Creates **downloadable PDF and CSV reports** for maintenance teams
- Displays a **10-page enterprise Streamlit dashboard**

---

## ⚠️ Problem Statement

Elevators are critical infrastructure used by millions daily in:
- 🏥 Hospitals | ✈️ Airports | 🏢 Office Towers | 🛍️ Shopping Malls
- 🏨 Hotels | 🚉 Railway Stations | 🏗️ Apartments | 🏭 Industrial Facilities

**Current reactive maintenance** causes:
- Unexpected elevator breakdowns
- Passengers trapped inside elevators (safety risk)
- High emergency repair costs
- Extended downtime (hours to days)
- Reduced equipment lifespan

**Our Solution:** Predict failures **before** they occur using AI analysis of sensor data.

---

## 🏗️ System Architecture

```
📡 Sensor Data (16 parameters)
          │
          ▼
🧹 Data Processing (DataProcessor)
   • Missing value handling
   • Outlier detection (IQR × 3.0)
   • Ordinal encoding (Vibration, Brake, Bearing)
   • One-Hot encoding (Error Code → 8 binary features)
   • StandardScaler normalization
          │
          ▼
⚙️ Feature Engineering (FeatureEngineer)
   • +10 domain-specific features
   • Total: 32 features
          │
          ▼
🔄 SMOTE Balancing
   • Before: [8888, 28456, 2656] (imbalanced)
   • After:  [28456, 28456, 28456] (balanced)
          │
          ▼
🤖 ML Model Training (3 classifiers)
   • Decision Tree, Random Forest, Logistic Regression
   • 5-Fold Stratified Cross-Validation
   • Best Model Auto-Selection (by F1 Score)
          │
          ▼
💾 Model Persistence (Joblib)
   • saved_models/best_model.pkl
   • saved_models/preprocessor.pkl
   • saved_models/feature_engineer.pkl
          │
          ▼
🖥️ Streamlit Application (10 pages)
   • Prediction • Dashboard • Analytics
   • Reports • Explainable AI • and more
```

---

## 🗃️ Dataset

| Property | Value |
|---|---|
| **File** | `elevator_predictive_maintenance_research_50000.csv` |
| **Records** | 50,000 elevator readings |
| **Features** | 17 columns (16 features + 1 target) |
| **Missing Values** | None |
| **Duplicates** | None |
| **Target** | Status (3 classes) |

### Class Distribution (Severe Imbalance)

| Class | Count | Percentage |
|---|---|---|
| Maintenance Required | 35,570 | 71.1% |
| Healthy | 11,110 | 22.2% |
| Failure Predicted | 3,320 | 6.6% |

> **Solution:** SMOTE (Synthetic Minority Over-sampling Technique) applied to training data.

### Feature Descriptions

| Feature | Type | Range | Encoding |
|---|---|---|---|
| Elevator_ID | ID | EL000001–EL050000 | Dropped |
| Motor_Temperature | Numerical | 25–95 °C | StandardScaler |
| Ambient_Temperature | Numerical | 18–42 °C | StandardScaler |
| Humidity | Numerical | 25–90 % | StandardScaler |
| Vibration_Level | Ordinal | Low/Medium/High/Very High | Ordinal (0–3) |
| Motor_Current_A | Numerical | 4–25 A | StandardScaler |
| Power_Consumption_kW | Numerical | 1.5–18 kW | StandardScaler |
| Running_Hours | Numerical | 0–25,000 hrs | StandardScaler |
| Door_Open_Count | Numerical | 0–150,000 | StandardScaler |
| Load_Weight | Numerical | 0–1,200 kg | StandardScaler |
| Cabin_Speed_mps | Numerical | 0.5–2.5 m/s | StandardScaler |
| Brake_Condition | Ordinal | Good/Fair/Poor | Ordinal (0–2) |
| Bearing_Condition | Ordinal | Good/Fair/Poor | Ordinal (0–2) |
| Last_Maintenance_Days | Numerical | 0–365 days | StandardScaler |
| Sensor_Health_Score | Numerical | 50–100 | StandardScaler |
| Error_Code | Nominal | E000/E101–E601 | One-Hot (8 features) |
| **Status (Target)** | Target | 3 classes | Label Encoding (0,1,2) |

---

## ⚙️ Feature Engineering

10 advanced features derived using elevator engineering domain knowledge:

| Feature | Formula | Range | Purpose |
|---|---|---|---|
| **Motor_Stress_Index** | `(Motor_Temp/95) × (Motor_Current/25) × 100` | 0–100 | Combined thermal + electrical stress |
| **Failure_Risk_Score** | `0.30×Vib + 0.25×Brake + 0.25×Bearing + 0.20×(1-SensorHealth) × 100` | 0–100 | Weighted risk of imminent failure |
| **Maintenance_Risk_Score** | `min(LastMaintDays/365, 1) × 100` | 0–100 | Risk from overdue maintenance |
| **Door_Usage_Index** | `Door_Open_Count / max(Running_Hours, 1)` | 0–∞ | Door wear rate (actuations/hour) |
| **Sensor_Reliability_Index** | `Sensor_Health_Score / 100` | 0–1 | Data trustworthiness |
| **Power_Efficiency** | `Power_kW / max(Load_kg, 1) × 1000` | W/kg | Mechanical efficiency indicator |
| **Operating_Efficiency** | `(Speed/2.5) × (1 - Vibration/3) × 100` | 0–100 | Speed vs vibration efficiency |
| **Health_Score** | `100 - 0.50×FRS - 0.30×MRS - 0.20×MSI` | 0–100 | Overall composite health |
| **Mechanical_Wear_Index** | `(Running_Hours / 25000) × 100` | 0–100 | % of rated motor life consumed |
| **Environmental_Stress_Index** | `(Humidity-25)/65 × 0.5 + (AmbTemp-18)/24 × 0.5` | 0–1 | Environmental harshness |

---

## 🤖 Machine Learning Models

| Model | Description | Config |
|---|---|---|
| Decision Tree | Simple interpretable model | max_depth=12, balanced |
| Random Forest | 200-tree ensemble (**Best**) | n_estimators=200, balanced |
| Logistic Regression | Linear multinomial classifier | C=1.0, max_iter=1000 |

### Training Strategy
- **Train/Test Split:** 80/20 stratified
- **SMOTE:** Applied to training data only (prevents data leakage)
- **Cross-Validation:** 5-fold stratified
- **Selection Criterion:** Highest weighted F1 Score

---

## 📊 Model Results

| Model | Accuracy | F1 Score | ROC-AUC | CV Mean | Train Time |
|---|---|---|---|---|---|
| ⭐ **Random Forest** | **1.0000** | **1.0000** | **1.0000** | **1.0000** | 17.1s |
| Logistic Regression | 0.9985 | 0.9985 | 1.0000 | 0.9984 | 60.5s |
| Decision Tree | 0.9922 | 0.9924 | 0.9994 | 0.9914 | 3.1s |

> **Note:** Perfect accuracy reflects the high quality of the engineered features and the clear separability of the three health states in the dataset.

---

## 🖥️ Application Pages

| Page | Description |
|---|---|
| 🏠 **Home** | Project overview, problem statement, workflow, benefits |
| 🔮 **Prediction** | Single elevator prediction with full analysis |
| 📊 **Dashboard** | System-wide KPI cards and interactive charts |
| 📈 **Analytics** | Deep-dive analysis across all features |
| 🗃️ **Dataset Explorer** | Searchable, filterable dataset browser |
| 🤖 **Model Performance** | All 3 model comparison, confusion matrices |
| 📄 **Reports** | PDF and CSV report generation |
| 🧠 **Explainable AI** | SHAP waterfall, summary, and force plots |
| ⚙️ **Settings** | Configurable thresholds and model settings |
| ℹ️ **About** | Project methodology and architecture |

---

## 🚀 Installation

### Prerequisites
- Python 3.11+
- pip

### Step 1: Clone Repository
```bash
git clone https://github.com/your-username/elevator-predictive-maintenance.git
cd elevator-predictive-maintenance
```

### Step 2: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 3: Train the Model
```bash
python models/train_model.py
```

Expected output:
```
🎉 TRAINING COMPLETE in ~360 seconds
🏆 Best Model: Random Forest
   Accuracy:  1.0000
   F1 Score:  1.0000
   ROC-AUC:   1.0000
```

### Step 4: Launch the Application
```bash
streamlit run app.py
```

Open your browser at: `http://localhost:8501`

---

## 📱 Usage

### Making a Prediction

1. Navigate to the **🔮 Prediction** page
2. Enter sensor readings (Motor Temperature, Vibration Level, Brake/Bearing condition, etc.)
3. Click **"Predict Elevator Health"**
4. View:
   - Prediction class (Healthy / Maintenance Required / Failure Predicted)
   - Confidence score and probability breakdown
   - Risk gauge and meter
   - Remaining Useful Life estimate
   - Color-coded alerts
   - Priority-ordered maintenance recommendations
5. Download PDF or JSON report

### Exploring Analytics

1. Navigate to **📈 Analytics** page
2. Use the 6 analysis tabs for comprehensive data exploration
3. Filter by status, condition, or error code

### Understanding AI Decisions

1. Navigate to **🧠 Explainable AI** page
2. View SHAP global feature importance
3. Generate waterfall plots for individual predictions
4. Read plain-language prediction explanations

---

## ☁️ Deployment

### Streamlit Community Cloud

1. Push your code to a GitHub repository
2. Go to [streamlit.io/cloud](https://streamlit.io/cloud)
3. Click **"New app"** and connect your repository
4. Set **Main file path:** `app.py`
5. Click **Deploy**

### Environment Files

| File | Purpose |
|---|---|
| `requirements.txt` | Python package dependencies |
| `runtime.txt` | Python version specification |

| `.gitignore` | Files excluded from version control |

> **Note:** The trained model files (`saved_models/*.pkl`) should be committed to the repository for cloud deployment, or retrained on first deployment by running the training script.

---

## 📁 Project Structure

```
Predictive Maintenance of elevators using ml/
├── 📁 dataset/
│   └── elevator_predictive_maintenance_research_50000.csv
├── 📁 models/
│   └── train_model.py          # Complete ML training pipeline
├── 📁 saved_models/            # Trained model artifacts (created after training)
│   ├── best_model.pkl
│   ├── preprocessor.pkl
│   ├── feature_engineer.pkl
│   ├── label_encoder.pkl
│   ├── model_results.pkl
│   └── full_results.pkl
├── 📁 utils/
│   ├── __init__.py
│   ├── data_processor.py       # DataProcessor class
│   ├── feature_engineer.py     # FeatureEngineer class
│   ├── model_trainer.py        # ModelTrainer class
│   ├── predictor.py            # ElevatorPredictor class
│   ├── alert_system.py         # AlertSystem class
│   ├── recommendation_engine.py # RecommendationEngine class
│   ├── report_generator.py     # ReportGenerator class
│   └── visualizations.py      # Visualizations class
├── 📁 pages/
│   ├── 01_Prediction.py
│   ├── 02_Dashboard.py
│   ├── 03_Analytics.py
│   ├── 04_Dataset_Explorer.py
│   ├── 05_Model_Performance.py
│   ├── 06_Reports.py
│   ├── 07_Explainable_AI.py
│   ├── 08_Settings.py
│   └── 09_About.py
├── 📁 assets/
│   └── css/
│       └── styles.css          # Custom CSS design system
├── 📁 reports/                 # Generated reports
├── 📁 notebooks/               # Jupyter notebooks
├── 🐍 app.py                   # Main Streamlit application (Home)
├── 📋 requirements.txt
├── 📋 runtime.txt
├── 📋 packages.txt
├── 📋 .gitignore
└── 📖 README.md
```

---

## 🛠️ Technologies Used

| Category | Technology | Version |
|---|---|---|
| **ML Framework** | Scikit-learn | 1.5.0 |
| **Class Balancing** | imbalanced-learn (SMOTE) | 0.12.3 |
| **Explainability** | SHAP | 0.45.1 |
| **Web Framework** | Streamlit | 1.35.0 |
| **Data Processing** | Pandas | 2.2.2 |
| **Numerical** | NumPy | 1.26.4 |
| **Visualization** | Plotly | 5.22.0 |
| **Visualization** | Matplotlib / Seaborn | 3.9.0 / 0.13.2 |
| **Model Persistence** | Joblib | 1.4.2 |
| **PDF Generation** | fpdf2 | 2.7.9 |

---

## 📜 License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgements

- Dataset sourced from elevator maintenance research studies
- SHAP library by Lundberg & Lee (2017) for model explainability
- Streamlit for the web application framework
- Scikit-learn team for the ML libraries

---



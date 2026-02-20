# Automated Data Science Agent 🚀

An end-to-end **prompt-driven Data Science Agent** that automates Exploratory Data Analysis (EDA), anomaly detection, feature understanding, and model recommendation.  
The system integrates **multi-agent analysis**, **AutoML**, and **ScaleDown-based metadata compression** to reduce analysis time from hours to minutes.

---

## 📌 Project Overview

Modern data analysis is time-consuming and repetitive. This project introduces an **intelligent data science agent** that:

- Accepts datasets and natural-language prompts
- Automatically performs EDA
- Detects anomalies
- Recommends machine learning models
- Compresses dataset metadata using ScaleDown
- Generates structured insights and reports
- Provides an interactive, professional dashboard

The goal is to **boost data scientist productivity** while maintaining analytical quality.

---

## ✨ Key Features

- **Prompt-driven EDA**  
  Describe what you want to analyze in natural language.

- **Automated Exploratory Data Analysis**
  - Dataset profiling
  - Missing value analysis
  - Statistical summaries

- **Anomaly Detection**
  - Isolation Forest–based outlier detection on numeric features

- **AutoML Model Recommendation**
  - Automatic task inference (classification / regression)
  - Baseline Random Forest model suggestion
  - Performance metrics

- **ScaleDown Integration**
  - Compresses dataset schema and statistics by ~75%
  - Enables fast reasoning on large datasets
  - Preserves structural relationships

- **Interactive Frontend**
  - Dataset upload (CSV / Parquet)
  - Target column selection
  - Visual analytics dashboard
  - Professional UI inspired by React design

---

## 🧠 System Architecture

User Prompt + Dataset
↓
Streamlit Frontend
↓
FastAPI Backend
↓
Multi-Agent Orchestrator
┌───────────────┐
│ Profiling │
│ Visualization │
│ Insight │
│ Anomaly │
│ AutoML │
└───────────────┘
↓
ScaleDown Compression
↓
Insights + Models + Reports


---

## 🛠️ Tech Stack

### Backend
- Python 3.10+
- FastAPI
- Pandas
- Scikit-learn
- DuckDB
- Pydantic v2
- Pydantic Settings

### Frontend
- Streamlit
- Plotly

### Intelligence & Analytics
- Isolation Forest (Anomaly Detection)
- Random Forest (AutoML Baseline)
- ScaleDown (Metadata Compression)

---

## 📂 Project Structure

data_science_agent/
│
├── app/
│ ├── main.py
│ ├── config.py
│ ├── agents/
│ ├── ingestion/
│ ├── scaledown/
│ ├── reports/
│ └── utils/
│
├── frontend/
│ └── streamlit_app.py
│
├── datasets/
├── reports/
├── requirements.txt
└── README.md


---

## ⚙️ Installation & Setup

### Create Virtual Environment

```bash
python -m venv DAagent

Activate it: Windows

DAagent\Scripts\activate

Linux / macOS

source DAagent/bin/activate

Install Dependencies
pip install -r requirements.txt

Configure Environment Variables

Create a .env file in the project root:

SCALEDOWN_API_KEY=your_api_key_here
HOST=127.0.0.1
PORT=8000

Running the Application
Start Backend (FastAPI)
uvicorn app.main:app --host 127.0.0.1 --port 8000


Verify backend:

http://127.0.0.1:8000/docs

Start Frontend (Streamlit)
streamlit run frontend/streamlit_app.py


Access UI:

http://localhost:8501

How to Use

Upload a CSV or Parquet dataset

Enter an analysis prompt

(Optional) Select a target column for AutoML

Click Run Analysis

View:

Insights

Anomalies

Visualizations

Model recommendations

ScaleDown compression metrics



Author

Abhiroop Mukherjee
Data Science & Machine Learning Enthusiast
Project: Automated Data Science Agent
# Flipkart Sentiment Analysis – MLflow & Prefect MLOps 🚀

This repository demonstrates an **end-to-end MLOps workflow** for training, tracking, registering, and scheduling a Sentiment Analysis model using **MLflow** and **Prefect**.

The project extends a Flipkart Product Review Sentiment Analysis use case with **industry-ready MLOps practices**.

---

## 🧠 Project Overview

- Multiple ML models trained for sentiment classification
- Experiments tracked using **MLflow**
- Metrics & hyperparameters visualized
- Best model registered and tagged
- Training pipeline orchestrated and scheduled using **Prefect**

---

## 🛠️ Tech Stack

| Category | Tools |
|--------|------|
| Language | Python |
| ML | scikit-learn |
| Experiment Tracking | MLflow |
| Orchestration | Prefect |
| Visualization | MLflow UI, Prefect UI |
| Storage | Local (can be extended to S3 / DB) |

---

## 📁 Repository Structure

flipkart-sentiment-mlflow/
│
├── data/
│ └── data.csv
│
├── mlops/
│ └── screenshots/
│ ├── mlflow/
│ │ ├── 01_experiment_runs.png
│ │ ├── 02_run_params_metrics.png
│ │ ├── 03_metric_plots.png
│ │ ├── 04_hyperparameter_plots.png
│ │ ├── 05_model_registry.png
│ │ ├── 06_model_tags.png
│ │ └── 07_run_Params_Metrics_Artifacts.png
│ └── prefect/
│
├── notebooks/
├── scripts/
├── prefect_flow.py
├── README.md
└── .gitignore


---

## 🔍 MLflow – Experiment Tracking

### ✔ Experiment Runs
Tracks multiple model runs with different:
- Algorithms (Logistic Regression, SVM, Random Forest)
- Vectorizers (BoW, TF-IDF)
- Hyperparameters

📸  
![MLflow Runs](mlops/screenshots/mlflow/01_experiment_runs.png)

---

### ✔ Parameters & Metrics
- Logged parameters: `model`, `feature`, `max_features`
- Logged metrics: `accuracy`, `f1_score`

📸  
![Params & Metrics](mlops/screenshots/mlflow/02_run_params_metrics.png)

---

### ✔ Metric Plots
Visual comparison of:
- Accuracy vs F1-score across runs

📸  
![Metric Plots](mlops/screenshots/mlflow/03_metric_plots.png)

---

### ✔ Hyperparameter Visualization
Parallel coordinate plots for hyperparameter tuning

📸  
![Hyperparameter Plots](mlops/screenshots/mlflow/04_hyperparameter_plots.png)

---

## 📦 Model Registry

### ✔ Model Registration
Best-performing model registered in MLflow Model Registry

📸  
![Model Registry](mlops/screenshots/mlflow/05_model_registry.png)

---

### ✔ Model Tagging
Models tagged for:
- Algorithm
- Feature type
- Owner
- Stage (production)
- Use case

📸  
![Model Tags](mlops/screenshots/mlflow/06_model_tags.png)

---

### ✔ Model Artifacts
Stored artifacts include:
- Model pickle
- Environment files
- Requirements

📸  
![Artifacts](mlops/screenshots/mlflow/07_run_Params_Metrics_Artifacts.png)

---

## ⏱️ Prefect – Workflow Orchestration

### ✔ Training Pipeline
Prefect flow handles:
- Data loading
- Model training
- Metric logging to MLflow

### ✔ Scheduling
- Flow deployed using Prefect
- Scheduled to run automatically (hourly)
- Monitored via Prefect UI

---

## 🚀 How to Run Locally

```bash
conda activate flipkart-mlflow-env
python prefect_flow.py

🎯 Key Highlights

✅ End-to-end MLOps workflow
✅ MLflow experiment tracking & model registry
✅ Prefect scheduling and orchestration
✅ Industry-aligned project structure

👩‍💻 Author

Revathy Gopinath
🔗 GitHub: https://github.com/revathygopinath
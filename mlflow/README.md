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

## 📂 Repository Structure

```text
flipkart-sentiment-mlflow/
│
├── data/
│   └── data.csv
│
├── mlops/
│   └── screenshots/
│       ├── mlflow/
│       │   ├── 01_experiment_runs.png
│       │   ├── 02_run_params_metrics.png
│       │   ├── 03_metric_plots.png
│       │   ├── 04_hyperparameter_plots.png
│       │   ├── 05_model_registry.png
│       │   ├── 06_model_tags.png
│       │   └── 07_run_Params_Metrics_Artifacts.png
│       │
│       └── prefect/
│           ├── 01_flow_graph.png
│           ├── 02_flow_run_completed.png
│           └── 03_deployment.png
│
├── notebooks/
│   └── EDA_Modelling_Flipkart_Product_Reviews_MLflow.ipynb
│
├── scripts/
│
├── prefect_flow.py
├── README.md
└── .gitignore


---
## 🔍 MLflow – Experiment Tracking

### 📊 Experiment Runs
Tracks multiple model runs across different algorithms, vectorizers, and hyperparameters.

![MLflow Experiment Runs](mlops/screenshots/mlflow/01_experiment_runs.png)




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

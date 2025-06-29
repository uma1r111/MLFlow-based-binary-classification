# MLflow Binary Classification Tracking (with DagsHub Deployment)

This project demonstrates how to use **MLflow** to track machine learning experiments on an imbalanced binary classification problem using multiple models and sampling techniques.

It includes:
- A baseline logistic regression model
- Random Forest and XGBoost models
- Handling data imbalance with **SMOTETomek**
- Logging and tracking metrics, parameters, and models with **MLflow**
- (Optional) Cloud-based experiment tracking using **DagsHub**

---

## 🎯 Project Goal

The objective was to understand and demonstrate:

- How to track experiments across different ML models and data preprocessing steps.
- How to use **MLflow** for logging metrics, hyperparameters, and models.
- How to **version and promote models** (e.g., from dev to prod).
- How to optionally deploy and view experiment runs via **DagsHub**, a Git-integrated MLflow backend.

---

## 🧪 Models Trained

| Model                        | Notes                                |
|-----------------------------|--------------------------------------|
| Logistic Regression         | Baseline model                       |
| Random Forest               | Tree-based ensemble model            |
| XGBoost                     | Boosting model for better accuracy   |
| XGBoost with SMOTETomek     | Balanced dataset handling            |

---

## 📦 Running Locally

1. **Create virtual environment**
```bash
conda create -n mlflow-env python=3.10
conda activate mlflow-env
```
2. **Install dependencies**
```bash
pip install -r requirements.txt
```
3. **Start MLflow tracking server (optional)**
```bash
mlflow ui
```
4. **Cloud Logging via DagsHub**
```bash
import mlflow
mlflow.set_tracking_uri("https://dagshub.com/<username>/<repo_name>.mlflow")
mlflow.set_experiment("Anomaly-Detection")
```
5. **Configure Credentials**
```bash
import os
os.environ['MLFLOW_TRACKING_USERNAME'] = 'your-dagshub-username'
os.environ['MLFLOW_TRACKING_PASSWORD'] = 'your-dagshub-token'
```
6. **Deployment**
```bash
# Promoted to Production
client = mlflow.MlflowClient()
client.copy_model_version(src_model_uri, dst_name="model-prod")

# Loaded for inference
loaded_model = mlflow.sklearn.load_model("models:/<model_name>/<version>")
```

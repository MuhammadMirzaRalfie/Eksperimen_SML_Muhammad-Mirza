import pandas as pd
import numpy as np
import os
import sys
import argparse
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Setup Arguments
parser = argparse.ArgumentParser()
parser.add_argument("--n_estimators", type=int, default=100)
parser.add_argument("--max_depth", type=int, default=None)
args = parser.parse_args()

# Setup DagsHub
DAGSHUB_REPO_OWNER = "MuhammadMirzaRalfie"
DAGSHUB_REPO_NAME = "Eksperimen_SML_Mirza"
remote_server_uri = f"https://dagshub.com/{DAGSHUB_REPO_OWNER}/{DAGSHUB_REPO_NAME}.mlflow"

mlflow.set_tracking_uri(remote_server_uri)

if "MLFLOW_TRACKING_USERNAME" not in os.environ:
    try:
        import dagshub
        dagshub.init(repo_owner=DAGSHUB_REPO_OWNER, repo_name=DAGSHUB_REPO_NAME, mlflow=True)
    except ImportError:
        pass

# Load Data
try:
    df = pd.read_csv('train_preprocessing.csv')
except FileNotFoundError:
    sys.exit(1)

X = df.drop('y', axis=1)
y = df['y']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Training
mlflow.set_experiment("Eksperimen_RF_Automated")

with mlflow.start_run() as run:
    mlflow.log_param("n_estimators", args.n_estimators)
    mlflow.log_param("max_depth", args.max_depth)

    model = RandomForestClassifier(
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        random_state=42
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    
    mlflow.log_metric("accuracy", acc)
    mlflow.sklearn.log_model(model, "model")

    # Important: Print Run ID for CI/CD capture
    print(f"[INFO] Run ID saved: {run.info.run_id}")
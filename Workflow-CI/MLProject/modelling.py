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


parser = argparse.ArgumentParser()
parser.add_argument("--n_estimators", type=int, default=100, help="Jumlah pohon di Random Forest")
parser.add_argument("--max_depth", type=int, default=None, help="Kedalaman maksimum pohon")
args = parser.parse_args()

DAGSHUB_REPO_OWNER = "MuhammadMirzaRalfie"
DAGSHUB_REPO_NAME = "Eksperimen_SML_Mirza"


remote_server_uri = f"https://dagshub.com/{DAGSHUB_REPO_OWNER}/{DAGSHUB_REPO_NAME}.mlflow"
mlflow.set_tracking_uri(remote_server_uri)

if "MLFLOW_TRACKING_USERNAME" not in os.environ:
    print("[INFO] Tidak mendeteksi Env Var login, mencoba load manual/dagshub init...")
    import dagshub
    dagshub.init(repo_owner=DAGSHUB_REPO_OWNER, repo_name=DAGSHUB_REPO_NAME, mlflow=True)


print("[INFO] Loading Data...")
try:

    df = pd.read_csv('train_preprocessing.csv')
except FileNotFoundError:
    print("Error: File 'train_preprocessing.csv' tidak ditemukan.")
    sys.exit(1)

X = df.drop('y', axis=1)
y = df['y']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)


print(f"[INFO] Training dengan n_estimators={args.n_estimators}, max_depth={args.max_depth}")

mlflow.set_experiment("Eksperimen_RF_Automated")

with mlflow.start_run():
    
    mlflow.log_param("n_estimators", args.n_estimators)
    mlflow.log_param("max_depth", args.max_depth)

    
    model = RandomForestClassifier(
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        random_state=42
    )
    
    model.fit(X_train, y_train)
    
    # Evaluasi
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    
    print(f"Accuracy: {acc}")
    
    # Log Metric & Model
    mlflow.log_metric("accuracy", acc)
    mlflow.sklearn.log_model(model, "model")
    
    print("[SUCCESS] Training selesai & tercatat di DagsHub.")
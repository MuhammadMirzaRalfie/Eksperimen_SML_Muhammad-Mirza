import pandas as pd
import numpy as np
import os
import sys
import argparse
import mlflow
import mlflow.sklearn
import yaml
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# --- 1. SETUP ARGUMENTS ---
parser = argparse.ArgumentParser()
parser.add_argument("--n_estimators", type=int, default=100)
parser.add_argument("--max_depth", type=int, default=None)
args = parser.parse_args()

# --- 2. SETUP DAGSHUB ---
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

# --- 3. LOAD DATA (Robust Path) ---
# Menggunakan path absolut agar file selalu ketemu dimanapun script dijalankan
current_dir = os.path.dirname(os.path.abspath(__file__))
csv_path = os.path.join(current_dir, 'train_preprocessing.csv')

print(f"[INFO] Loading Data from: {csv_path}")
try:
    df = pd.read_csv(csv_path)
except FileNotFoundError:
    print(f"[ERROR] File csv tidak ditemukan di {csv_path}")
    sys.exit(1)

X = df.drop('y', axis=1)
y = df['y']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# --- 4. TRAINING ---
mlflow.set_experiment("Eksperimen_RF_Automated")

# --- PERBAIKAN UTAMA: LOAD CONDA.YAML DENGAN ABSOLUTE PATH ---
conda_env_path = os.path.join(current_dir, "conda.yaml")
conda_env = None

if os.path.exists(conda_env_path):
    print(f"[INFO] Conda Env ditemukan di: {conda_env_path}")
    with open(conda_env_path, "r") as f:
        conda_env = yaml.safe_load(f)
        # Pastikan python versi benar terdeteksi
        print("[INFO] Isi Conda Env (Partial):", str(conda_env)[:100])
else:
    print(f"[WARNING] FILE CONDA.YAML TIDAK DITEMUKAN DI {conda_env_path}!")
    print("[WARNING] MLflow akan menggunakan Default Environment (Risiko Python 3.8!)")

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
    
    # Tempelkan conda_env yang berisi Python 3.9 ke dalam model
    print("[INFO] Logging model dengan Environment khusus...")
    mlflow.sklearn.log_model(model, "model", conda_env=conda_env)

    # Simpan Run ID
    run_id = run.info.run_id
    print(f"[INFO] Run ID saved: {run_id}")
    
    # Tulis ke file (gunakan path absolut juga agar aman)
    run_id_file = os.path.join(current_dir, "last_run_id.txt") # Opsional, jika script main.yml pakai grep, ini tidak wajib tapi bagus.
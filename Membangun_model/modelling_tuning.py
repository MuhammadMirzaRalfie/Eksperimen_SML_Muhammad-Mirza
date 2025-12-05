import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import dagshub

DAGSHUB_REPO_NAME = "Eksperimen_SML_Mirza"
FILENAME = 'train_preprocessing.csv'

try:
    dagshub.init(repo_owner=DAGSHUB_REPO_OWNER, repo_name=DAGSHUB_REPO_NAME, mlflow=True)
except Exception as e:
    print(f"[WARNING] Gagal init DagsHub: {e}")

# 2. LOAD DATA
print("[INFO] Loading data...")
if not os.path.exists(FILENAME):
    raise FileNotFoundError(f"File '{FILENAME}' tidak ditemukan. Pastikan file ada di folder yang sama.")

df = pd.read_csv(FILENAME) 
X = df.drop('y', axis=1)
y = df['y']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print("[INFO] Starting Grid Search...")

# 3. TRAINING (GRID SEARCH)
rf = RandomForestClassifier(random_state=42)
param_grid = {
    'n_estimators': [50, 100],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5]
}

grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=3, n_jobs=-1, verbose=2)
grid_search.fit(X_train, y_train)

best_model = grid_search.best_estimator_
best_params = grid_search.best_params_

print(f"[INFO] Best Params: {best_params}")

# 4. MLFLOW LOGGING
mlflow.set_experiment("Eksperimen_RF_Tuning_Mirza")

print("[INFO] Memulai logging ke MLflow...")
with mlflow.start_run(run_name="RandomForest_Tuning_Best") as run:
    
    # A. LOG PARAMETERS
    print("[INFO] Logging parameters...")
    mlflow.log_params(best_params)
    
    # B. LOG METRICS
    print("[INFO] Logging metrics...")
    y_pred = best_model.predict(X_test)
    
    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, average='weighted'),
        "recall": recall_score(y_test, y_pred, average='weighted'),
        "f1_score": f1_score(y_test, y_pred, average='weighted')
    }
    mlflow.log_metrics(metrics)
    
    # C. LOG MODEL (Dengan Error Handling)
    print("[INFO] Logging Model...")
    try:
        mlflow.sklearn.log_model(best_model, "model_random_forest")
    except Exception as e:
        print(f"[WARNING] Gagal upload model: {e}")

    # D. LOG ARTIFACTS (Dengan Error Handling)
    print("[INFO] Generating & Logging Artifacts...")
    
    try:
        # Artefak 1: Confusion Matrix Image
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(6,5))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title('Confusion Matrix')
        plt.ylabel('Actual')
        plt.xlabel('Predicted')
        plt.savefig("confusion_matrix.png")
        plt.close()
        mlflow.log_artifact("confusion_matrix.png")
        
        # Artefak 2: Feature Importance Image
        if hasattr(best_model, 'feature_importances_'):
            importances = best_model.feature_importances_
            indices = np.argsort(importances)[::-1][:10] # Top 10
            
            plt.figure(figsize=(10,6))
            plt.title("Top 10 Feature Importances")
            plt.bar(range(len(indices)), importances[indices], align="center")
            plt.xticks(range(len(indices)), X.columns[indices], rotation=45)
            plt.tight_layout()
            plt.savefig("feature_importance.png")
            plt.close()
            mlflow.log_artifact("feature_importance.png")
            
        print("[SUCCESS] Artifacts berhasil diupload.")
        
    except Exception as e:
        
        print(f"[WARNING] Gagal upload artifacts: {e}")

    print(f"[DONE] Run selesai. Cek DagsHub: https://dagshub.com/{DAGSHUB_REPO_OWNER}/{DAGSHUB_REPO_NAME}")

# Cleanup file lokal
for f in ["confusion_matrix.png", "feature_importance.png"]:
    if os.path.exists(f):
        os.remove(f)
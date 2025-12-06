import argparse
import os
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib
import mlflow

def main(data_path):
    mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", "./mlruns"))
    mlflow.set_experiment("workflow-ci")

    # Load dataset
    data = pd.read_csv(data_path)
    
    # Dummy preprocessing: ambil semua kolom numerik sebagai fitur
    X = data.select_dtypes(include=["float64", "int64"])
    y = data["target"] if "target" in data.columns else pd.Series([0]*len(data))

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    with mlflow.start_run():
        # Train model
        model = LogisticRegression(max_iter=1000)
        model.fit(X_train, y_train)

        # Predict & log metrics
        preds = model.predict(X_test)
        acc = accuracy_score(y_test, preds)
        mlflow.log_metric("accuracy", acc)

        # Log model
        mlflow.sklearn.log_model(model, artifact_path="model")

        print(f"Model trained with accuracy: {acc}")
        print(f"Model artifacts saved to: mlruns/<run_id>/artifacts/model")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="train_preprocessing.csv")
    args = parser.parse_args()
    main(args.data_path)

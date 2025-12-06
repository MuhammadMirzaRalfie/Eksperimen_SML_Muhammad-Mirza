import argparse
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import mlflow
import mlflow.sklearn

def main(args):
    # Set tracking URI lokal agar konsisten di GitHub Actions
    mlflow.set_tracking_uri("mlruns")
    mlflow.set_experiment("training-experiment")

    # Optional tapi recommended
    mlflow.sklearn.autolog()

    df = pd.read_csv(args.data_path)

    X = df.drop(args.target, axis=1)
    y = df[args.target]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    with mlflow.start_run(run_name="training") as run:
        model = RandomForestClassifier(
            n_estimators=50,
            random_state=42
        )

        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)

        mlflow.log_metric("accuracy_manual", acc)

        os.makedirs(args.model_output, exist_ok=True)

        mlflow.sklearn.save_model(
            sk_model=model,
            path=args.model_output
        )

        print(f"Training completed. Model saved at: {args.model_output}")
        print("MLflow Run ID:", run.info.run_id)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--target", type=str, default="y")
    parser.add_argument("--model_output", type=str, required=True)
    args = parser.parse_args()
    main(args)

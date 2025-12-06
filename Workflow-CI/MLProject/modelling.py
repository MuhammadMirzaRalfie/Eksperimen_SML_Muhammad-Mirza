import argparse
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import mlflow
import mlflow.sklearn

def main(args):
    # Load dataset
    df = pd.read_csv(args.data_path)

    X = df.drop(args.target, axis=1)
    y = df[args.target]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    mlflow.sklearn.autolog()

    with mlflow.start_run(run_name="Auto_Retrain"):
        model = RandomForestClassifier(n_estimators=50, random_state=42)
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        mlflow.log_metric("accuracy_manual", acc)

        # Pastikan folder output ada
        os.makedirs(args.model_output, exist_ok=True)

        # Simpan model ke MLflow format
        mlflow.sklearn.save_model(
            sk_model=model,
            path=args.model_output
        )

        print(f"Model saved to: {args.model_output}")
        print(f"Accuracy: {acc}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--data_path",
        type=str,
        required=True,
        help="Path ke dataset preprocessing"
    )

    parser.add_argument(
        "--target",
        type=str,
        default="y",
        help="Nama kolom target"
    )

    parser.add_argument(
        "--model_output",
        type=str,
        required=True,
        help="Folder output untuk menyimpan model MLflow"
    )

    args = parser.parse_args()
    main(args)

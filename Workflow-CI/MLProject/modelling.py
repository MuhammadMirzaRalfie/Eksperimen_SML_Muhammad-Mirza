import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import argparse
import os


def load_data(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Dataset tidak ditemukan: {path}")
    df = pd.read_csv(path)
    return df


def train_model(df, target_col):
    X = df.drop(columns=[target_col])
    y = df[target_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        random_state=42
    )

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    return model, acc


def main(args):
    mlflow.set_experiment("workflow-ci-model")

    with mlflow.start_run():
        df = load_data(args.data_path)

        mlflow.log_param("target", args.target)
        mlflow.log_param("dataset_rows", len(df))

        model, acc = train_model(df, args.target)

        mlflow.log_metric("accuracy", acc)

        mlflow.sklearn.log_model(model, artifact_path="model")

        print(f"Akurasi model: {acc:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Training Model CI MLflow")
    parser.add_argument(
        "--data_path",
        type=str,
        default="train_preprocessing.csv",
        help="Path dataset hasil preprocessing"
    )
    parser.add_argument(
        "--target",
        type=str,
        default="y", 
        help="Nama kolom target"
    )

    args = parser.parse_args()
    main(args)

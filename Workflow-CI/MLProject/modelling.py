import argparse
import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


def load_data(path):
    df = pd.read_csv(path)

    # Cek kolom y ada atau tidak
    if "y" not in df.columns:
        raise ValueError("Kolom 'y' tidak ditemukan dalam dataset.")

    return df


def train_model(df):
    # Pisahkan fitur dan target
    X = df.drop("y", axis=1)
    y = df["y"]

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Model
    model = RandomForestClassifier(
        n_estimators=150,
        random_state=42,
        n_jobs=-1
    )

    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)

    return model, acc


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, required=True)
    args = parser.parse_args()

    df = load_data(args.data_path)

    with mlflow.start_run():
        model, acc = train_model(df)

        # Log metrics
        mlflow.log_metric("accuracy", acc)

        # Log model
        mlflow.sklearn.log_model(model, artifact_path="model")

        print("Training completed. Accuracy:", acc)

import argparse
import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


def main(args):
    # ensure no active runs (fix CI issue)
    mlflow.end_run()

    # set experiment
    mlflow.set_experiment("training-experiment")

    # load data
    df = pd.read_csv(args.data_path)

    X = df.drop(columns=[args.target])
    y = df[args.target]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    with mlflow.start_run(run_name="training"):
        # model
        model = RandomForestClassifier(
            n_estimators=100,
            random_state=42
        )

        model.fit(X_train, y_train)
        preds = model.predict(X_test)

        acc = accuracy_score(y_test, preds)
        mlflow.log_metric("accuracy", acc)

        # log model
        mlflow.sklearn.log_model(model, args.model_output)

        print(f"Training completed. Accuracy: {acc}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--target", type=str, required=True)
    parser.add_argument("--model_output", type=str, default="model_output")

    args = parser.parse_args()
    main(args)

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import mlflow
import mlflow.sklearn

# Load Data
try:
    df = pd.read_csv('train_preprocessing.csv')
except FileNotFoundError:
    print("Error: File 'train_preprocessing.csv' tidak ditemukan.")
    exit()

X = df.drop('y', axis=1)
y = df['y']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


mlflow.set_tracking_uri("http://127.0.0.1:5000") 

mlflow.sklearn.autolog()

with mlflow.start_run(run_name="Basic_Model"):
    model = RandomForestClassifier(n_estimators=10, random_state=42)
    

    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {acc}")
    

    mlflow.log_metric("accuracy_manual", acc)
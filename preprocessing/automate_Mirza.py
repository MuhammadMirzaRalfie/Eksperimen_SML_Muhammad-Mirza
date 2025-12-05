import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer


BASE_DIR = os.path.dirname(os.path.abspath(__file__))

INPUT_PATH = os.path.join(BASE_DIR, '..', 'train_raw', 'train.csv')

OUTPUT_PATH = os.path.join(BASE_DIR, 'train_preprocessing.csv')

# Konfigurasi Data
EDU_MAP = {'primary': 1, 'secondary': 2, 'tertiary': 3}
OUTLIER_COLS = ['age', 'balance', 'duration', 'campaign']
TARGET_COL = 'y'

def cap_outliers(df):
    df_clean = df.copy()
    exist_cols = [c for c in OUTLIER_COLS if c in df_clean.columns]
    for col in exist_cols:
        Q1 = df_clean[col].quantile(0.25)
        Q3 = df_clean[col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        df_clean[col] = np.clip(df_clean[col], lower, upper)
    return df_clean

def basic_cleaning(df):
    df_clean = df.copy()
    df_clean.replace('unknown', np.nan, inplace=True)
    if 'education' in df_clean.columns:
        df_clean['education'] = df_clean['education'].map(EDU_MAP)
    return df_clean

def run_pipeline():
    print("="*40)
    print("   STARTING PIPELINE (MIRZA STRUCTURE)   ")
    print("="*40)

    # 1. Load Data
    if not os.path.exists(INPUT_PATH):
        print(f"[ERROR] Input file tidak ditemukan di: {INPUT_PATH}")
        print("Pastikan nama file di folder train_raw adalah 'train.csv'")
        return

    df = pd.read_csv(INPUT_PATH)
    print(f"[1] Data Loaded. Shape: {df.shape}")

    # 2. Proses Data
    df = basic_cleaning(df)
    df = cap_outliers(df)

    # Pisahkan Target
    target_data = None
    if TARGET_COL in df.columns:
        le = LabelEncoder()
        target_data = le.fit_transform(df[TARGET_COL])
        df = df.drop(columns=[TARGET_COL])

    # 3. Pipeline Transformasi
    num_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    cat_cols = df.select_dtypes(include=['object']).columns.tolist()

    num_pipe = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    cat_pipe = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])

    preprocessor = ColumnTransformer([
        ('num', num_pipe, num_cols),
        ('cat', cat_pipe, cat_cols)
    ])

    X_matrix = preprocessor.fit_transform(df)

    # 4. Rapikan Output
    try:
        ohe_features = preprocessor.named_transformers_['cat']['onehot'].get_feature_names_out(cat_cols)
        feature_names = num_cols + list(ohe_features)
        df_result = pd.DataFrame(X_matrix, columns=feature_names)
    except:
        df_result = pd.DataFrame(X_matrix)

    if target_data is not None:
        df_result[TARGET_COL] = target_data

    # 5. Simpan
    df_result.to_csv(OUTPUT_PATH, index=False)
    print(f"[SUCCESS] Data disimpan di: {OUTPUT_PATH}")

if __name__ == "__main__":
    run_pipeline()
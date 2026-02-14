import pandas as pd
import numpy as np

from sklearn.preprocessing import (
    StandardScaler, MinMaxScaler,
    RobustScaler, MaxAbsScaler,
    LabelEncoder
)

from sklearn.impute import KNNImputer
import category_encoders as ce

def preprocess_data(df, target, cleaning, encoding, normalization):
    if target not in df.columns:
        raise ValueError("Target column not found")

    # -------------------------
    # Handle missing values
    # -------------------------
    if cleaning == "drop_rows":
        df = df.dropna()
    elif cleaning == "drop_columns":
        df = df.dropna(axis=1)
    elif cleaning == "mean":
        df = df.fillna(df.mean(numeric_only=True))
    elif cleaning == "median":
        df = df.fillna(df.median(numeric_only=True))
    elif cleaning == "mode":
        df = df.fillna(df.mode().iloc[0])
    elif cleaning == "ffill":
        df = df.ffill()
    elif cleaning == "bfill":
        df = df.bfill()
    elif cleaning == "knn":
        # KNN Imputer only works on numeric data usually, or needs encoding first.
        # But standard sklearn KNNImputer is for numeric. 
        # For simplicity, let's apply it to numeric columns only and fill others with mode?
        # Or better, just apply to numeric columns.
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if not numeric_cols.empty:
            imputer = KNNImputer(n_neighbors=5)
            df[numeric_cols] = imputer.fit_transform(df[numeric_cols])
        # Fill remaining non-numeric NaNs with mode if any
        df = df.fillna(df.mode().iloc[0])

    y = df[target]
    X = df.drop(columns=[target])

    # -------------------------
    # Encoding
    # -------------------------
    if encoding == "one_hot":
        X = pd.get_dummies(X)
    elif encoding == "label":
        for col in X.select_dtypes(include=["object"]).columns:
            X[col] = LabelEncoder().fit_transform(X[col])
    elif encoding == "ordinal":
        for col in X.select_dtypes(include=["object"]).columns:
            X[col] = LabelEncoder().fit_transform(X[col])
    elif encoding == "binary":
        encoder = ce.BinaryEncoder()
        X = encoder.fit_transform(X, y)
    elif encoding == "frequency":
        encoder = ce.CountEncoder(normalize=True)
        X = encoder.fit_transform(X, y)
    elif encoding == "target":
        encoder = ce.TargetEncoder()
        X = encoder.fit_transform(X, y)
    elif encoding == "hashing":
        encoder = ce.HashingEncoder()
        X = encoder.fit_transform(X, y)

    # -------------------------
    # Normalization
    # -------------------------
    scaler = None

    if normalization == "zscore":
        scaler = StandardScaler()
    elif normalization == "min_max":
        scaler = MinMaxScaler()
    elif normalization == "robust":
        scaler = RobustScaler()
    elif normalization == "maxabs":
        scaler = MaxAbsScaler()
    elif normalization == "log":
        X = np.log1p(X)
    elif normalization == "none":
        scaler = None

    if scaler:
        X = pd.DataFrame(
            scaler.fit_transform(X),
            columns=X.columns
        )

    return X, y

from fastapi import FastAPI, HTTPException
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel
import pandas as pd
from code_generator import generate_model_code
import numpy as np
import os, time, joblib
import category_encoders as ce
from sklearn.impute import KNNImputer

from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, MaxAbsScaler, LabelEncoder, QuantileTransformer, PowerTransformer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score


app = FastAPI(title="ML Training Service")

# -------------------------
# Allowed options
# -------------------------
ALGORITHMS = [
  "logistic", "knn", "svm",
  "random_forest", "gradient_boosting",
  "linear_regression"
]

ENCODINGS = [
  "one_hot", "label", "ordinal", "binary", "frequency", "target", "hashing"
]

NORMALIZATIONS = [
  "zscore", "min_max", "robust", "maxabs",
  "log", "power_transform", "quantile", "none"
]

CLEANING_STRATEGIES = [
  "drop_rows", "drop_columns", "mean", "median",
  "mode", "constant", "ffill", "bfill", "knn", "interpolation"
]


# -------------------------
# Request schema
# -------------------------
class MLRequest(BaseModel):
    dataset_path: str
    cleaning_strategy: str
    encoding_method: str
    normalization_technique: str
    algorithm: str
    target_column: str


class CodeGenerationRequest(BaseModel):
    target_column: str
    algorithm: str
    cleaning_strategy: str
    encoding_method: str
    normalization_technique: str


# -------------------------
# Endpoint
# -------------------------
@app.post("/train")
def train(request: MLRequest):
    # -------------------------
    # Step 1: Validate inputs
    # -------------------------
    if request.algorithm.lower() not in ALGORITHMS:
        raise HTTPException(status_code=400, detail=f"Unsupported algorithm: {request.algorithm}. Supported: {ALGORITHMS}")

    if request.encoding_method.lower() not in ENCODINGS:
        raise HTTPException(status_code=400, detail=f"Unsupported encoding method: {request.encoding_method}. Supported: {ENCODINGS}")

    if request.normalization_technique.lower() not in NORMALIZATIONS:
        raise HTTPException(status_code=400, detail=f"Unsupported normalization technique: {request.normalization_technique}. Supported: {NORMALIZATIONS}")

    if request.cleaning_strategy.lower() not in CLEANING_STRATEGIES:
        raise HTTPException(status_code=400, detail=f"Unsupported cleaning strategy: {request.cleaning_strategy}. Supported: {CLEANING_STRATEGIES}")

    # -------------------------
    # Step 2: Load dataset
    # -------------------------
    try:
        df = pd.read_csv(request.dataset_path)
        df.columns = df.columns.str.strip()
    except FileNotFoundError:
        raise HTTPException(status_code=400, detail=f"Dataset not found at path: {request.dataset_path}")
    except pd.errors.EmptyDataError:
        raise HTTPException(status_code=400, detail="Dataset is empty")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to read dataset: {str(e)}")

    # -------------------------
    # Step 3: Validate target
    # -------------------------
    target_col = request.target_column.strip()
    if target_col not in df.columns:
        raise HTTPException(
            status_code=400,
            detail=f"Target column '{target_col}' not found in dataset. Columns: {df.columns.tolist()}"
        )

    # -------------------------
    # Step 4: Preprocess
    # -------------------------
    try:
        X, y = preprocess_data(
            df=df,
            target=target_col,
            cleaning=request.cleaning_strategy.lower(),
            encoding=request.encoding_method.lower(),
            normalization=request.normalization_technique.lower()
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Preprocessing failed: {str(e)}")

    # -------------------------
    # Step 5: Train model
    # -------------------------
    try:
        result = train_model(X, y, request.algorithm.lower())
        return result
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=f"Algorithm error: {str(ve)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Model training failed: {str(e)}")


# -------------------------
# Preprocessing function
# -------------------------
def preprocess_data(df, target, cleaning, encoding, normalization):
    # Handle missing values
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
    elif cleaning == "constant":
        df = df.fillna(0)
    elif cleaning == "ffill":
        df = df.ffill()
    elif cleaning == "bfill":
        df = df.bfill()
    elif cleaning == "interpolation":
        df = df.interpolate()
    elif cleaning == "knn":
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if not numeric_cols.empty:
            imputer = KNNImputer(n_neighbors=5)
            df[numeric_cols] = imputer.fit_transform(df[numeric_cols])
        # Fill remaining non-numeric NaNs with mode if any
        df = df.fillna(df.mode().iloc[0])

    if target not in df.columns:
        raise ValueError("Target column not found after cleaning")

    y = df[target]
    X = df.drop(columns=[target])

    # Encoding
    if encoding == "one_hot":
        X = pd.get_dummies(X)
    elif encoding in ["label", "ordinal"]:
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

    # Normalization
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
    elif normalization == "power_transform":
        scaler = PowerTransformer(method='yeo-johnson')
    elif normalization == "quantile":
        scaler = QuantileTransformer(output_distribution='normal')
    elif normalization == "none":
        scaler = None

    if scaler:
        X = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

    return X, y


def is_regression_target(y):
    return pd.api.types.is_numeric_dtype(y) and y.nunique() > 10


# -------------------------
# Training function
# -------------------------
def train_model(X, y, algorithm):
    is_regression = is_regression_target(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # -------------------------
    # REGRESSION
    # -------------------------
    if is_regression:
        if algorithm == "linear_regression":
            model = LinearRegression()
        elif algorithm == "random_forest":
            model = RandomForestRegressor()
        elif algorithm == "gradient_boosting":
            model = GradientBoostingRegressor()
        else:
            raise ValueError(f"{algorithm} not supported for regression")

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        metrics = {
            "mse": mse,
            "rmse": rmse,
            "r2_score": r2_score(y_test, y_pred)
        }

        problem_type = "regression"

    # -------------------------
    # CLASSIFICATION
    # -------------------------
    else:
        if algorithm == "logistic":
            model = LogisticRegression(max_iter=1000)
        elif algorithm == "knn":
            model = KNeighborsClassifier()
        elif algorithm == "svm":
            model = SVC()
        elif algorithm == "random_forest":
            model = RandomForestClassifier()
        elif algorithm == "gradient_boosting":
            model = GradientBoostingClassifier()
        else:
            raise ValueError(f"{algorithm} not supported for classification")

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        metrics = {
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred, average="weighted", zero_division=0),
            "recall": recall_score(y_test, y_pred, average="weighted", zero_division=0),
            "f1_score": f1_score(y_test, y_pred, average="weighted", zero_division=0)
        }

        problem_type = "classification"

    os.makedirs("models", exist_ok=True)
    model_path = f"models/{algorithm}_{int(time.time())}.joblib"
    joblib.dump(model, model_path)

    return {
        "problem_type": problem_type,
        **metrics,
        "model_path": os.path.abspath(model_path)
    }


# -------------------------
# Code Generation Endpoint
# -------------------------
@app.post("/generate-code", response_class=PlainTextResponse)
def generate_code(request: CodeGenerationRequest):
    """
    Generate Python code from model configuration.
    Returns the code as plain text for download.
    """

    print("generating the code")
    try:
        code = generate_model_code(
            target_column=request.target_column,
            algorithm=request.algorithm,
            cleaning_strategy=request.cleaning_strategy,
            encoding_method=request.encoding_method,
            normalization_technique=request.normalization_technique
        )
        return code
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Code generation failed: {str(e)}")

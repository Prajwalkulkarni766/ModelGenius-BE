"""
Code Generator Module
Generates Python code from model configuration for export.
"""


def generate_model_code(
    target_column: str,
    algorithm: str,
    cleaning_strategy: str,
    encoding_method: str,
    normalization_technique: str,
) -> str:
    """
    Generate a complete Python script based on model configuration.
    
    Args:
        target_column: Name of the target column
        algorithm: ML algorithm used (logistic, knn, svm, random_forest, etc.)
        cleaning_strategy: Missing value handling strategy
        encoding_method: Categorical encoding method
        normalization_technique: Normalization/scaling technique
    
    Returns:
        Complete Python script as a string
    """
    
    # Map algorithm names to sklearn class names
    algorithm_class_map = {
        "logistic": "LogisticRegression",
        "knn": "KNeighborsClassifier",
        "svm": "SVC",
        "random_forest": "RandomForestClassifier",
        "gradient_boosting": "GradientBoostingClassifier",
        "linear_regression": "LinearRegression",
    }
    
    alg_class = algorithm_class_map.get(algorithm, "RandomForestClassifier")
    
    # Check if it's a regression algorithm for imports
    is_regression = algorithm == "linear_regression"
    
    code = f'''import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, LabelEncoder, MaxAbsScaler
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, mean_squared_error

# 1. Load Data
# Replace 'dataset.csv' with your actual file path
df = pd.read_csv('dataset.csv')

# 2. Preprocessing
# Target Column
target_col = "{target_column}"
if target_col not in df.columns:
    raise ValueError(f"Target column {{target_col}} not found")

# Handling Missing Values: {cleaning_strategy}
'''
    
    # Add cleaning strategy code
    if cleaning_strategy == "drop_rows":
        code += 'df = df.dropna()\n'
    elif cleaning_strategy == "drop_columns":
        code += 'df = df.dropna(axis=1)\n'
    elif cleaning_strategy == "mean":
        code += 'df = df.fillna(df.mean(numeric_only=True))\n'
    elif cleaning_strategy == "median":
        code += 'df = df.fillna(df.median(numeric_only=True))\n'
    elif cleaning_strategy == "mode":
        code += 'df = df.fillna(df.mode().iloc[0])\n'
    elif cleaning_strategy == "constant":
        code += 'df = df.fillna(0)\n'
    elif cleaning_strategy == "ffill":
        code += 'df = df.ffill()\n'
    elif cleaning_strategy == "bfill":
        code += 'df = df.bfill()\n'
    elif cleaning_strategy == "knn":
        code += '''from sklearn.impute import KNNImputer
numeric_cols = df.select_dtypes(include=[np.number]).columns
imputer = KNNImputer(n_neighbors=5)
df[numeric_cols] = imputer.fit_transform(df[numeric_cols])
df = df.fillna(df.mode().iloc[0])
'''
    elif cleaning_strategy == "interpolation":
        code += 'df = df.interpolate()\n'
    
    code += f'''
# Split X and y
X = df.drop(columns=[target_col])
y = df[target_col]

# Encoding: {encoding_method}
'''
    
    # Add encoding code
    if encoding_method == "one_hot":
        code += 'X = pd.get_dummies(X)\n'
    elif encoding_method in ["label", "ordinal"]:
        code += '''for col in X.select_dtypes(include=["object"]).columns:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col])
'''
    elif encoding_method == "binary":
        code += '''import category_encoders as ce
encoder = ce.BinaryEncoder()
X = encoder.fit_transform(X, y)
'''
    elif encoding_method == "frequency":
        code += '''import category_encoders as ce
encoder = ce.CountEncoder(normalize=True)
X = encoder.fit_transform(X, y)
'''
    elif encoding_method == "target":
        code += '''import category_encoders as ce
encoder = ce.TargetEncoder()
X = encoder.fit_transform(X, y)
'''
    elif encoding_method == "hashing":
        code += '''import category_encoders as ce
encoder = ce.HashingEncoder()
X = encoder.fit_transform(X, y)
'''
    
    code += f'''
# Normalization: {normalization_technique}
scaler = None
'''
    
    # Add normalization code
    if normalization_technique == "min_max":
        code += 'scaler = MinMaxScaler()\n'
    elif normalization_technique == "zscore":
        code += 'scaler = StandardScaler()\n'
    elif normalization_technique == "robust":
        code += 'scaler = RobustScaler()\n'
    elif normalization_technique == "maxabs":
        code += 'scaler = MaxAbsScaler()\n'
    elif normalization_technique == "log":
        code += 'X = np.log1p(X)\n'
    elif normalization_technique == "power_transform":
        code += '''from sklearn.preprocessing import PowerTransformer
scaler = PowerTransformer(method='yeo-johnson')
'''
    elif normalization_technique == "quantile":
        code += '''from sklearn.preprocessing import QuantileTransformer
scaler = QuantileTransformer(output_distribution='normal')
'''
    
    code += '''
if scaler:
    X = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

# 3. Train Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

'''
    
    code += f'''# 4. Model Training: {algorithm}
model = {alg_class}()
model.fit(X_train, y_train)

# 5. Evaluation
predictions = model.predict(X_test)
'''
    
    if is_regression:
        code += '''print("MSE:", mean_squared_error(y_test, predictions))
'''
    else:
        code += '''try:
    print("Accuracy:", accuracy_score(y_test, predictions))
except:
    print("MSE:", mean_squared_error(y_test, predictions))
'''
    
    return code

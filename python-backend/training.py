from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

import joblib, os, time

def train_model(X, y, algorithm):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # -------------------------
    # Model selection
    # -------------------------
    algorithm = algorithm.lower().strip()

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
        raise ValueError(f"Unsupported algorithm: '{algorithm}'")

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # -------------------------
    # Metrics
    # -------------------------
    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, average="weighted", zero_division=0),
        "recall": recall_score(y_test, y_pred, average="weighted", zero_division=0),
        "f1_score": f1_score(y_test, y_pred, average="weighted", zero_division=0)
    }

    # -------------------------
    # Save model
    # -------------------------
    os.makedirs("models", exist_ok=True)
    model_path = f"models/{algorithm}_{int(time.time())}.joblib"
    joblib.dump(model, model_path)

    return {
        **metrics,
        "model_path": model_path
    }

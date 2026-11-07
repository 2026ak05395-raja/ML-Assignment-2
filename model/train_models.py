"""Train 5 classifiers on the Breast Cancer Wisconsin (Diagnostic) dataset,
evaluate them on a held-out test split and persist models + artifacts."""

from pathlib import Path

import joblib
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

RANDOM_STATE = 42
MODEL_DIR = Path(__file__).resolve().parent
ROOT_DIR = MODEL_DIR.parent
TARGET = "target"


def load_data() -> pd.DataFrame:
    bunch = load_breast_cancer(as_frame=True)
    df = bunch.frame.copy()
    # sklearn encodes 0 = malignant, 1 = benign; flip so malignant (positive class) = 1
    df[TARGET] = 1 - df[TARGET]
    return df


def build_models() -> dict:
    def scaled(clf):
        return Pipeline([("scaler", StandardScaler()), ("clf", clf)])

    return {
        "Logistic Regression": scaled(LogisticRegression(max_iter=5000, random_state=RANDOM_STATE)),
        "Decision Tree": DecisionTreeClassifier(max_depth=5, random_state=RANDOM_STATE),
        "KNN": scaled(KNeighborsClassifier(n_neighbors=7)),
        "Naive Bayes": GaussianNB(),
        "SVM": scaled(CalibratedClassifierCV(SVC(kernel="rbf", random_state=RANDOM_STATE), ensemble=False)),
    }


def evaluate(model, X, y) -> dict:
    y_pred = model.predict(X)
    y_prob = model.predict_proba(X)[:, 1]
    return {
        "Accuracy": accuracy_score(y, y_pred),
        "AUC": roc_auc_score(y, y_prob),
        "Precision": precision_score(y, y_pred),
        "Recall": recall_score(y, y_pred),
        "F1": f1_score(y, y_pred),
        "MCC": matthews_corrcoef(y, y_pred),
    }


def main():
    df = load_data()
    X, y = df.drop(columns=[TARGET]), df[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )

    pd.concat([X_test, y_test], axis=1).to_csv(ROOT_DIR / "test_data.csv", index=False)
    joblib.dump(list(X.columns), MODEL_DIR / "feature_names.joblib")

    rows = []
    for name, model in build_models().items():
        model.fit(X_train, y_train)
        joblib.dump(model, MODEL_DIR / f"{name.lower().replace(' ', '_')}.joblib")
        rows.append({"Model": name, **evaluate(model, X_test, y_test)})

    results = pd.DataFrame(rows).set_index("Model").round(4)
    results.to_csv(MODEL_DIR / "metrics.csv")
    print(results.to_string())


if __name__ == "__main__":
    main()

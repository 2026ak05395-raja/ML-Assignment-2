"""Streamlit app: evaluate pre-trained breast cancer classifiers on an uploaded test CSV."""

from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import streamlit as st
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)

MODEL_DIR = Path(__file__).resolve().parent / "model"
TARGET = "target"
MODEL_FILES = {
    "Logistic Regression": "logistic_regression.joblib",
    "Decision Tree": "decision_tree.joblib",
    "KNN": "knn.joblib",
    "Naive Bayes": "naive_bayes.joblib",
    "SVM": "svm.joblib",
}

st.set_page_config(page_title="Breast Cancer Classifier Evaluation", layout="wide")


@st.cache_resource
def load_model(name: str):
    return joblib.load(MODEL_DIR / MODEL_FILES[name])


@st.cache_resource
def load_feature_names() -> list:
    return joblib.load(MODEL_DIR / "feature_names.joblib")


def compute_metrics(y_true, y_pred, y_prob) -> dict:
    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "AUC": roc_auc_score(y_true, y_prob) if y_true.nunique() > 1 else float("nan"),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall": recall_score(y_true, y_pred, zero_division=0),
        "F1": f1_score(y_true, y_pred, zero_division=0),
        "MCC": matthews_corrcoef(y_true, y_pred),
    }


def plot_confusion_matrix(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    fig, ax = plt.subplots(figsize=(4, 3.5))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax,
        xticklabels=["Benign (0)", "Malignant (1)"],
        yticklabels=["Benign (0)", "Malignant (1)"],
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    return fig


st.title("🩺 Breast Cancer Diagnosis – Model Evaluation")
st.caption(
    "Upload a test CSV (30 features + `target`, where 1 = malignant, 0 = benign). "
    "Models are pre-trained; uploaded data is used only for evaluation."
)

with st.sidebar:
    st.header("Settings")
    model_name = st.selectbox("Select model", list(MODEL_FILES))
    uploaded = st.file_uploader("Upload test data (CSV)", type="csv")

if uploaded is None:
    st.info("Upload a CSV file from the sidebar to begin (e.g. `test_data.csv` from the repo).")
    st.stop()

df = pd.read_csv(uploaded)
features = load_feature_names()
missing = [c for c in features + [TARGET] if c not in df.columns]
if missing:
    st.error(f"Uploaded CSV is missing required columns: {missing}")
    st.stop()

X, y = df[features], df[TARGET].astype(int)
st.write(f"Loaded **{len(df)}** rows × **{len(features)}** features.")
with st.expander("Preview uploaded data"):
    st.dataframe(df.head(20))

model = load_model(model_name)
y_pred = model.predict(X)
y_prob = model.predict_proba(X)[:, 1]
metrics = compute_metrics(y, y_pred, y_prob)

st.subheader(f"Evaluation metrics – {model_name}")
for col, (label, value) in zip(st.columns(len(metrics)), metrics.items()):
    col.metric(label, f"{value:.4f}")

left, right = st.columns(2)
with left:
    st.subheader("Confusion matrix")
    st.pyplot(plot_confusion_matrix(y, y_pred))
with right:
    st.subheader("Classification report")
    report = classification_report(
        y, y_pred, labels=[0, 1], target_names=["Benign", "Malignant"],
        output_dict=True, zero_division=0,
    )
    st.dataframe(pd.DataFrame(report).T.round(4))

st.subheader("Comparison across all models")
rows = []
for name in MODEL_FILES:
    m = load_model(name)
    rows.append({"Model": name, **compute_metrics(y, m.predict(X), m.predict_proba(X)[:, 1])})
comparison = pd.DataFrame(rows).set_index("Model").round(4)
st.dataframe(comparison.style.highlight_max(axis=0, color="#c6efce"))

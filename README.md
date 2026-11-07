# Breast Cancer Diagnosis – ML Classification & Streamlit Deployment

**Live app:** `<your-streamlit-app-url>.streamlit.app`
**GitHub repo:** https://github.com/2026ak05395-raja/ML-Assignment-2

## a. Problem Statement

Predict whether a breast tumour is **malignant (1)** or **benign (0)** from 30 numeric features that
describe cell nuclei in a digitised fine-needle aspirate (FNA) image. Five classification models are
trained and compared on six evaluation metrics, and the best model is deployed in an interactive
Streamlit web application.

## b. Dataset Description

| Item | Details |
|------|---------|
| Name | Breast Cancer Wisconsin (Diagnostic) |
| Source | UCI Machine Learning Repository (also bundled in `sklearn.datasets.load_breast_cancer`) |
| Instances | 569 (meets the ≥ 500 requirement) |
| Features | 30 numeric, real-valued (meets the ≥ 12 requirement) |
| Target | `target`: 1 = Malignant (212), 0 = Benign (357), binary |
| Missing values | None |

Features: ten cell-nucleus characteristics (radius, texture, perimeter, area, smoothness,
compactness, concavity, concave points, symmetry, fractal dimension). Each one is recorded three
ways: **mean**, **standard error** and **worst** (largest), which gives 30 features in total.

**Preprocessing / split:** stratified 80/20 train-test split (`random_state=42`), so the test set
has 114 rows: 42 malignant and 72 benign. The `StandardScaler` is fitted only on the training data.
It is stored inside the saved pipelines for the scale-sensitive models (LR, KNN, SVM). The held-out
test split is saved as `test_data.csv` so it can be uploaded to the app.

## c. Models Used & Evaluation Metrics

Models: Logistic Regression, Decision Tree (`max_depth=5`), K-Nearest Neighbours (`k=7`),
Gaussian Naive Bayes, and SVM (RBF kernel, probability-calibrated).

### Comparison Table (held-out test set, n = 114)

| ML Model Name       | Accuracy | AUC    | Precision | Recall | F1     | MCC    |
|---------------------|----------|--------|-----------|--------|--------|--------|
| Logistic Regression | 0.9649   | **0.9960** | 0.9750 | 0.9286 | 0.9512 | 0.9245 |
| Decision Tree       | 0.9211   | 0.9448 | 0.9459    | 0.8333 | 0.8861 | 0.8299 |
| KNN                 | 0.9561   | 0.9825 | 0.9744    | 0.9048 | 0.9383 | 0.9058 |
| Naive Bayes         | 0.9386   | 0.9934 | **1.0000** | 0.8333 | 0.9091 | 0.8715 |
| SVM                 | **0.9737** | 0.9947 | 0.9756  | **0.9524** | **0.9639** | **0.9433** |

### Observations

| ML Model Name | Observation about model performance |
|---------------|-------------------------------------|
| Logistic Regression | Strong linear baseline. It has the highest AUC (0.996), so its ranking of malignant vs benign is nearly perfect. This means the classes are close to linearly separable once the features are scaled. Recall (0.929) is a little behind SVM: 3 malignant cases are missed at the default 0.5 threshold. |
| Decision Tree | The weakest model on every metric (Acc 0.921, MCC 0.830). Recall is 0.833, so 7 of the 42 malignant tumours are missed. Its axis-aligned splits fit these correlated continuous features less well than the other models, and one tree has high variance. Ensembles such as Random Forest would reduce this. |
| KNN | Good overall (Acc 0.956, AUC 0.983) once the features are standardised. It is slightly below LR and SVM because distance-based voting is affected by the 30-dimensional, highly correlated feature space. Recall (0.905) is moderate. |
| Naive Bayes | Perfect precision (1.000): no false alarms. Recall is the joint lowest (0.833), so 7 malignant cases are missed. AUC is still high (0.993), so the probability ranking is good but the 0.5 threshold is badly calibrated, because the features are strongly correlated and break the independence assumption. |
| SVM | **Best overall model.** It has the highest Accuracy (0.974), Recall (0.952), F1 (0.964) and MCC (0.943), with AUC 0.995. The RBF kernel captures mild non-linearity in the margin. It misses only 2 malignant cases, which matters most in medical screening. |

**Conclusion:** SVM (RBF) is the recommended model. Logistic Regression is a close and more
interpretable second. Recall and MCC are the metrics to prioritise, because a false negative
(a missed malignancy) costs far more than a false positive.

## Streamlit App Features

1. **CSV upload**: upload test data containing the 30 features plus the `target` column.
2. **Model selection dropdown**: choose any of the 5 pre-trained models.
3. **Evaluation metrics**: Accuracy, AUC, Precision, Recall, F1, MCC.
4. **Confusion matrix & classification report** for the selected model.
5. **Comparison table** for all 5 models on the uploaded data, with the best value per metric highlighted.

The app only loads the saved `.joblib` models. It does not retrain on the uploaded data.

## Repository Structure

```
├── app.py                 # Streamlit application
├── requirements.txt       # Pinned dependencies
├── README.md
├── test_data.csv          # Held-out test split (upload this in the app)
└── model/
    ├── train_models.py    # Training + evaluation script
    ├── logistic_regression.joblib
    ├── decision_tree.joblib
    ├── knn.joblib
    ├── naive_bayes.joblib
    ├── svm.joblib
    ├── feature_names.joblib
    └── metrics.csv
```

## How to Run

```bash
pip install -r requirements.txt
python model/train_models.py   # retrains models, regenerates test_data.csv and metrics.csv
streamlit run app.py
```

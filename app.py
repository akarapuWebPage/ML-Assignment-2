"""Streamlit app for the Bank Marketing ML assignment.

Run from the project root:
    streamlit run app.py

Expected files:
    Model/logistic_regression.joblib
    Model/decision_tree.joblib
    Model/knn.joblib
    Model/gaussian_naive_bayes.joblib
    Model/svm.joblib
    Model/feature_information.joblib
    Model/model_comparison.csv
"""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    confusion_matrix,
    ConfusionMatrixDisplay,
    classification_report,
)


# ------------------------------------------------------------
# 1. PAGE CONFIGURATION
# ------------------------------------------------------------
st.set_page_config(
    page_title="Bank Marketing Classification",
    page_icon="📊",
    layout="wide",
)

PROJECT_DIR = (
    Path(__file__).resolve().parent
    if "__file__" in globals()
    else Path.cwd()
)
OUTPUT_DIR = PROJECT_DIR / "Model"

MODEL_FILES = {
    "Logistic Regression": "logistic_regression.joblib",
    "Decision Tree": "decision_tree.joblib",
    "KNN": "knn.joblib",
    "Gaussian Naive Bayes": "gaussian_naive_bayes.joblib",
    "SVM": "svm.joblib",
}

st.title("Bank Marketing Classification")
st.write(
    "Evaluate five trained classification models on an uploaded test CSV. "
    "The app loads saved pipelines and does not retrain them."
)
st.caption(
    "UCI Bank Marketing | 15 input features | Target: y (No / Yes)"
)


# ------------------------------------------------------------
# 2. LOAD SAVED ARTIFACTS
# ------------------------------------------------------------
@st.cache_resource
def load_saved_models():
    loaded = {}
    missing = []

    for model_name, filename in MODEL_FILES.items():
        model_path = OUTPUT_DIR / filename
        if model_path.exists():
            loaded[model_name] = joblib.load(model_path)
        else:
            missing.append(str(model_path.relative_to(PROJECT_DIR)))

    return loaded, missing


@st.cache_data
def load_training_comparison():
    results_path = OUTPUT_DIR / "model_comparison.csv"
    if results_path.exists():
        return pd.read_csv(results_path)
    return None


@st.cache_resource
def load_feature_information():
    info_path = OUTPUT_DIR / "feature_information.joblib"
    if info_path.exists():
        return joblib.load(info_path)
    return None


trained_models, missing_models = load_saved_models()
feature_info = load_feature_information()
training_results = load_training_comparison()

if missing_models:
    st.error(
        "Some saved model files are missing. Run the notebook's model-training "
        "and save-model sections, then place the generated files in `Model/`."
    )
    for item in missing_models:
        st.write(f"- `{item}`")
    st.stop()

if feature_info is None:
    st.error(
        "`Model/feature_information.joblib` was not found. "
        "Run the analysis notebook through its feature-information save section."
    )
    st.stop()

FEATURE_COLUMNS = feature_info["input_features"]
TARGET_COLUMN = feature_info.get("target", "y")


# ------------------------------------------------------------
# 3. SIDEBAR / NAVIGATION
# ------------------------------------------------------------
page = st.sidebar.radio(
    "Navigate",
    ["Project Overview", "Evaluate Uploaded Test Data", "Training Results"],
)


# ------------------------------------------------------------
# 4. PROJECT OVERVIEW
# ------------------------------------------------------------
if page == "Project Overview":
    st.header("Project Overview")
    st.write(
        "The objective is to predict whether a bank customer will subscribe "
        "to a term deposit. Five classification algorithms were trained on "
        "the original, imbalanced Bank Marketing dataset."
    )

    col1, col2, col3 = st.columns(3)
    col1.metric("Input features", len(FEATURE_COLUMNS))
    col2.metric("Saved models", len(trained_models))
    col3.metric("Target", TARGET_COLUMN)

    st.subheader("Methodology")
    st.markdown(
        """
        - Stratified 80:20 train-test split (`random_state=42`).
        - `duration` excluded to avoid data leakage.
        - Numerical features standardized; categorical features one-hot encoded.
        - Original class distribution retained; no resampling or class weighting.
        - Complete preprocessing and classifier pipelines saved together.
        """
    )

    st.subheader("Models included")
    st.write(", ".join(trained_models.keys()))

    st.info(
        "Use the **Evaluate Uploaded Test Data** page to upload a CSV containing "
        "the 15 predictor columns and the target column `y`."
    )


# ------------------------------------------------------------
# 5. EVALUATE UPLOADED TEST DATA
# ------------------------------------------------------------
elif page == "Evaluate Uploaded Test Data":
    st.header("Evaluate Models on Uploaded Test Data")
    st.write(
        "Upload a labelled test dataset. The app will evaluate the saved models "
        "on the same uploaded records without fitting or retraining any model."
    )

    uploaded_file = st.file_uploader(
        "Upload test data (CSV)",
        type=["csv"],
        help="CSV must contain all 15 input features and the target column y.",
    )

    if uploaded_file is not None:
        try:
            test_df = pd.read_csv(uploaded_file)
        except Exception as exc:
            st.error(f"Could not read the CSV file: {exc}")
            st.stop()

        # Normalize target values while preserving feature names.
        missing_features = [c for c in FEATURE_COLUMNS if c not in test_df.columns]
        if missing_features:
            st.error(
                "The uploaded CSV is missing these required input columns: "
                + ", ".join(missing_features)
            )
            st.stop()

        if TARGET_COLUMN not in test_df.columns:
            st.error(f"The uploaded CSV must also contain target column `{TARGET_COLUMN}`.")
            st.stop()

        if test_df.empty:
            st.error("The uploaded CSV contains no records.")
            st.stop()

        # Only the predictors used during training are passed to the pipelines.
        X_uploaded = test_df[FEATURE_COLUMNS].copy()
        raw_target = test_df[TARGET_COLUMN].astype(str).str.strip().str.lower()
        target_mapping = {
    **feature_info.get("target_mapping", {}),
    "no": 0,
    "yes": 1,
    "0": 0,
    "1": 1
}
        normalized_mapping = {str(k).strip().lower(): int(v) for k, v in target_mapping.items()}
        y_uploaded = raw_target.map(normalized_mapping)

        if y_uploaded.isna().any():
            unexpected = sorted(raw_target[y_uploaded.isna()].unique().tolist())
            st.error(
                f"Target column `{TARGET_COLUMN}` contains unsupported values: "
                f"{unexpected}. Expected 'no'/'yes' or their mapped numeric labels 0/1."
            )
            st.stop()

        y_uploaded = y_uploaded.astype(int)
        if set(y_uploaded.unique()) != {0, 1}:
            st.error(
                "The uploaded test data must contain both target classes (No and Yes) "
                "to calculate AUC and the full classification metrics."
            )
            st.stop()

        st.success(f"Loaded {len(test_df):,} test records successfully.")
        st.caption("Extra columns, if present (for example `duration`), are ignored.")
        with st.expander("Preview uploaded test data"):
            st.dataframe(test_df.head(10), use_container_width=True)

        selected_model = st.selectbox(
            "Select a model to inspect",
            list(trained_models.keys()),
        )

        if st.button("Evaluate all five models", type="primary"):
            evaluation_rows = []
            predictions_by_model = {}

            for model_name, model in trained_models.items():
                try:
                    predictions = model.predict(X_uploaded)

                    # Use probability scores when available; otherwise use
                    # the SVM decision function for ROC AUC.
                    if hasattr(model, "predict_proba"):
                        positive_scores = model.predict_proba(X_uploaded)[:, 1]
                    elif hasattr(model, "decision_function"):
                        positive_scores = model.decision_function(X_uploaded)
                    else:
                        positive_scores = predictions

                    metrics = {
                        "Model": model_name,
                        "Accuracy": accuracy_score(y_uploaded, predictions),
                        "AUC": roc_auc_score(y_uploaded, positive_scores),
                        "Precision": precision_score(y_uploaded, predictions, zero_division=0),
                        "Recall": recall_score(y_uploaded, predictions, zero_division=0),
                        "F1 Score": f1_score(y_uploaded, predictions, zero_division=0),
                        "MCC": matthews_corrcoef(y_uploaded, predictions),
                    }
                    evaluation_rows.append(metrics)
                    predictions_by_model[model_name] = predictions
                except Exception as exc:
                    st.error(f"Evaluation failed for {model_name}: {exc}")
                    st.stop()

            evaluation_df = pd.DataFrame(evaluation_rows)
            st.session_state["uploaded_evaluation"] = evaluation_df
            st.session_state["uploaded_predictions"] = predictions_by_model
            st.session_state["uploaded_y_true"] = y_uploaded
            st.session_state["uploaded_row_count"] = len(test_df)

        if "uploaded_evaluation" in st.session_state:
            evaluation_df = st.session_state["uploaded_evaluation"]
            predictions_by_model = st.session_state["uploaded_predictions"]
            y_true = st.session_state["uploaded_y_true"]

            st.subheader("Evaluation Metrics — Uploaded Test Data")
            st.dataframe(
                evaluation_df.set_index("Model").round(4),
                use_container_width=True,
            )

            st.subheader("Compare Models")
            metric_choice = st.selectbox(
                "Choose a metric to compare",
                ["Accuracy", "AUC", "Precision", "Recall", "F1 Score", "MCC"],
                key="uploaded_metric_choice",
            )
            chart_df = evaluation_df.set_index("Model")[[metric_choice]]
            st.bar_chart(chart_df)

            st.subheader(f"Detailed Results — {selected_model}")
            selected_predictions = predictions_by_model[selected_model]

            cm = confusion_matrix(y_true, selected_predictions, labels=[0, 1])
            fig, ax = plt.subplots(figsize=(5, 4))
            ConfusionMatrixDisplay(
                confusion_matrix=cm,
                display_labels=["No", "Yes"],
            ).plot(ax=ax, cmap="Blues", values_format="d", colorbar=False)
            ax.set_title(f"Confusion Matrix — {selected_model}")
            st.pyplot(fig)
            plt.close(fig)

            report = classification_report(
                y_true,
                selected_predictions,
                labels=[0, 1],
                target_names=["No", "Yes"],
                output_dict=True,
                zero_division=0,
            )
            st.subheader("Classification Report")
            st.dataframe(pd.DataFrame(report).transpose().round(4), use_container_width=True)

            csv_bytes = evaluation_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                "Download evaluation metrics (CSV)",
                data=csv_bytes,
                file_name="uploaded_test_evaluation.csv",
                mime="text/csv",
            )


# ------------------------------------------------------------
# 6. TRAINING RESULTS FROM ORIGINAL EXPERIMENT
# ------------------------------------------------------------
elif page == "Training Results":
    st.header("Original Training Experiment Results")
    st.write(
        "These are the results saved by the notebook from its original stratified "
        "80:20 experiment. They are separate from the evaluation of an uploaded CSV."
    )

    if training_results is None:
        st.warning("`Model/model_comparison.csv` was not found. Run the analysis notebook first.")
    else:
        expected_columns = [
            "Model", "Training Accuracy", "Testing Accuracy", "Accuracy Gap (pp)",
            "AUC", "Precision", "Recall", "F1 Score", "MCC",
        ]
        available_columns = [c for c in expected_columns if c in training_results.columns]
        st.dataframe(
            training_results[available_columns].round(4),
            use_container_width=True,
            hide_index=True,
        )

        if "Testing Accuracy" in training_results.columns:
            st.subheader("Testing Accuracy by Model")
            st.bar_chart(training_results.set_index("Model")[["Testing Accuracy"]])


# ------------------------------------------------------------
# 7. FOOTER
# ------------------------------------------------------------
st.divider()
st.caption("Bank Marketing Classification | Machine Learning Assignment")

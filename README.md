# Bank Marketing Classification — Machine Learning Assignment 2

## 1. Problem Statement

The objective of this project is to predict whether a bank customer will subscribe to a term deposit following a marketing campaign. Five machine learning classification algorithms are trained and evaluated using the same dataset and train-test split.

A Streamlit application is included to let users upload labelled test data, select a trained classifier, compare model metrics, and inspect a confusion matrix and classification report. The application loads the saved model pipelines and evaluates uploaded data without retraining.

## 2. Dataset Description

- **Dataset:** Bank Marketing (`bank-full.csv`)
- **Source:** UCI Machine Learning Repository
- **Task:** Binary classification
- **Target column:** `y` (`no` / `yes`)
- **Number of records:** 45,211
- **Input features used:** 15

The original dataset contains 17 columns: 16 predictor columns and the target. The `duration` feature is excluded because it represents the duration of a call and would not be known before the marketing call takes place. Including it could introduce data leakage.

The 15 input features used are:

- **Numerical features:** `age`, `balance`, `day`, `campaign`, `pdays`, `previous`
- **Categorical features:** `job`, `marital`, `education`, `default`, `housing`, `loan`, `contact`, `month`, `poutcome`

The target is encoded as `no = 0` and `yes = 1`.

### Preprocessing and train-test split

- A stratified 80:20 train-test split is used with `random_state=42`.
- Numerical features are standardized using `StandardScaler`.
- Categorical features are one-hot encoded using `OneHotEncoder(handle_unknown="ignore")`.
- The original class distribution is retained; no resampling or class weighting is applied.
- Preprocessing and each classifier are combined into a scikit-learn pipeline. The complete pipeline is saved for each model.

## 3. GitHub Repository Link

[ML Assignment 2 — GitHub Repository](https://github.com/akarapuWebPage/ML-Assignment-2)

## 4. Models Used and Evaluation

The following five classifiers are implemented on the same dataset:

1. Logistic Regression
2. Decision Tree Classifier
3. K-Nearest Neighbors (KNN)
4. Gaussian Naive Bayes
5. Support Vector Machine (SVM), implemented using `LinearSVC`

Each model is evaluated using the following metrics:

- **Accuracy:** Overall proportion of correct predictions.
- **AUC:** Area under the ROC curve; measures the model's ability to distinguish between the two classes.
- **Precision:** Proportion of predicted positive cases that are actually positive.
- **Recall:** Proportion of actual positive cases correctly identified.
- **F1 Score:** Harmonic mean of precision and recall.
- **MCC:** Matthews Correlation Coefficient; a balanced measure based on all four confusion-matrix outcomes.

### 4.1 Model comparison

The following results are taken from the completed experiment's `Model/model_comparison.csv`. Values are rounded to four decimal places.

| ML Model | Accuracy | AUC | Precision | Recall | F1 Score | MCC |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.8933 | 0.7717 | 0.6632 | 0.1786 | 0.2815 | 0.3065 |
| Decision Tree | 0.8696 | 0.6797 | 0.4184 | 0.2930 | 0.3446 | 0.2801 |
| KNN | 0.8874 | 0.7027 | 0.5529 | 0.1975 | 0.2911 | 0.2833 |
| Gaussian Naive Bayes | 0.8452 | 0.7514 | 0.3708 | 0.4641 | 0.4123 | 0.3271 |
| Support Vector Machine | 0.8932 | 0.7682 | 0.6513 | 0.1871 | 0.2907 | 0.3101 |

### 4.2 Observations on model performance

| ML Model | Observation |
|---|---|
| Logistic Regression | Achieved the highest accuracy (0.8933), AUC (0.7717), and precision (0.6632). However, its recall (0.1786) is low, meaning it identifies only a relatively small share of customers who subscribed. |
| Decision Tree | Achieved an F1 score of 0.3446 and recall of 0.2930. Its training accuracy (0.9277) is noticeably higher than its test accuracy (0.8696), suggesting some overfitting. It had the lowest AUC (0.6797) among the five models. |
| KNN | Achieved accuracy of 0.8874 and precision of 0.5529, but recall was 0.1975. Its F1 score (0.2911) and AUC (0.7027) were below those of Gaussian Naive Bayes and the leading models. |
| Gaussian Naive Bayes | Achieved the highest recall (0.4641), F1 score (0.4123), and MCC (0.3271). It identifies a larger share of actual subscribers, although its precision (0.3708) and accuracy (0.8452) are lower than those of the other models. |
| Support Vector Machine | Achieved accuracy of 0.8932 and precision of 0.6513, close to Logistic Regression. Its recall (0.1871) and F1 score (0.2907) were comparatively low, while its AUC was 0.7682. |

### Overall winner

**Gaussian Naive Bayes is the overall winner when considering the balance of positive-class detection and the reported F1 and MCC metrics.** It achieved the highest F1 score (0.4123), recall (0.4641), and MCC (0.3271) among the five models.

Logistic Regression achieved the highest accuracy, AUC, and precision, but its recall was substantially lower. Therefore, the choice of model depends on the objective: Gaussian Naive Bayes is preferable when identifying a larger share of potential subscribers is important, while Logistic Regression leads on accuracy, AUC, and precision.

## 5. Streamlit Application

The Streamlit application provides the following features:

- **Project Overview:** Displays the project objective, dataset information, modelling approach, and the five classifiers.
- **Test Data Upload:** Allows the user to upload a labelled CSV file containing the 15 input features and target column `y`.
- **Model Selection:** Allows the user to select a trained classifier for detailed evaluation.
- **Evaluation Metrics:** Displays Accuracy, AUC, Precision, Recall, F1 Score, and MCC.
- **Model Comparison:** Compares all five saved models on the uploaded test data.
- **Detailed Evaluation:** Displays a confusion matrix and classification report for the selected model.
- **Training Results:** Displays the original experiment results saved by the notebook in `Model/model_comparison.csv`.

The application loads the saved preprocessing-and-classifier pipelines. It does not train or retrain models using the uploaded test data.

## 6. Repository Structure

The repository is organized with the training notebook and all saved model artifacts inside the `Model/` directory.

```text
ML-Assignment-2/
├── app.py
├── README.md
├── requirements.txt
├── test_data.csv
└── Model/
    ├── ML_Assignment_2.ipynb
    ├── logistic_regression.joblib
    ├── decision_tree.joblib
    ├── knn.joblib
    ├── gaussian_naive_bayes.joblib
    ├── svm.joblib
    ├── feature_information.joblib
    └── model_comparison.csv
```

The files inside `Model/` serve the following purposes:

- `ML_Assignment_2.ipynb` contains the data analysis, preprocessing, model training, evaluation, and model-saving workflow.
- The five `.joblib` files contain the fitted preprocessing-and-classifier pipelines.
- `feature_information.joblib` stores the input feature names, numerical and categorical feature lists, target column, and target mapping.
- `model_comparison.csv` stores the training and test evaluation metrics for all five classifiers.

`test_data.csv` is the held-out test dataset exported by the notebook and used to demonstrate model evaluation in the Streamlit application.

## 7. Reproducibility Notes

- The train-test split is stratified and uses `random_state=42`.
- The same exported `test_data.csv` is used for the application demonstration.
- The original class distribution is retained during training.
- The application loads saved pipelines and does not train models on uploaded test data.
- The model artifacts were serialized using scikit-learn **1.6.1**. The dependency file pins this version to help maintain compatibility when loading the saved `.joblib` pipelines.

## 8. References

- UCI Machine Learning Repository — [Bank Marketing Dataset](https://archive.ics.uci.edu/dataset/222/bank+marketing)
- Streamlit Community Cloud — [https://streamlit.io/cloud](https://streamlit.io/cloud)
- Scikit-learn documentation — [https://scikit-learn.org/stable/](https://scikit-learn.org/stable/)

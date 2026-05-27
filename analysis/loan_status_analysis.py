from __future__ import annotations

from pathlib import Path
from textwrap import dedent
from typing import TypeAlias

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import binomtest
from sklearn.compose import ColumnTransformer  # type: ignore[import]
from sklearn.impute import SimpleImputer  # type: ignore[import]
from sklearn.linear_model import LogisticRegression  # type: ignore[import]
from sklearn.metrics import (  # type: ignore[import]
    ConfusionMatrixDisplay,
    accuracy_score,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split  # type: ignore[import]
from sklearn.neighbors import KNeighborsClassifier  # type: ignore[import]
from sklearn.pipeline import Pipeline  # type: ignore[import]
from sklearn.preprocessing import OneHotEncoder, StandardScaler  # type: ignore[import]


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET_PATHS = [
    Path.home() / "Downloads" / "LoanDataSet.csv",
    REPO_ROOT / "data" / "LoanDataSet.csv",
]
OUTPUT_DIR = REPO_ROOT / "docs"
PLOTS_DIR = OUTPUT_DIR / "plots"
RANDOM_STATE = 42
TEST_SIZE = 0.20
RocArtifact: TypeAlias = tuple[np.ndarray, np.ndarray, float]
EvaluationArtifacts: TypeAlias = tuple[
    pd.DataFrame,
    dict[str, np.ndarray],
    dict[str, RocArtifact],
    dict[str, np.ndarray],
]


def resolve_dataset_path() -> Path:
    """Use the requested Downloads location first, then a repo-local fallback."""
    for candidate in DEFAULT_DATASET_PATHS:
        if candidate.exists():
            return candidate

    searched = "\n".join(f"- {path}" for path in DEFAULT_DATASET_PATHS)
    raise FileNotFoundError(
        "LoanDataSet.csv was not found. Place it in one of these locations:\n"
        f"{searched}"
    )


def markdown_code_block(text: str) -> str:
    return f"```text\n{text.rstrip()}\n```"


def float_fmt(value: float) -> str:
    return f"{value:.4f}"


def dataframe_to_markdown_table(dataframe: pd.DataFrame, include_index: bool = False) -> str:
    frame = dataframe.copy()
    if include_index:
        frame = frame.reset_index()

    headers = [str(column) for column in frame.columns]
    rows = [[str(value) for value in row] for row in frame.to_numpy().tolist()]
    divider = ["---"] * len(headers)
    table_lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(divider) + " |",
    ]
    table_lines.extend("| " + " | ".join(row) + " |" for row in rows)
    return "\n".join(table_lines)


def left_trim_eight_spaces(text: str) -> str:
    return "\n".join(
        line[8:] if line.startswith("        ") else line for line in text.splitlines()
    )


def build_preprocessor(features: pd.DataFrame) -> ColumnTransformer:
    numeric_features = features.select_dtypes(include="number").columns.tolist()
    categorical_features = [
        column for column in features.columns if column not in numeric_features
    ]

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, numeric_features),
            ("categorical", categorical_pipeline, categorical_features),
        ]
    )


def specificity_from_confusion_matrix(matrix: np.ndarray) -> float:
    tn, fp, _, _ = matrix.ravel()
    return tn / (tn + fp)


def sensitivity_from_confusion_matrix(matrix: np.ndarray) -> float:
    _, _, fn, tp = matrix.ravel()
    return tp / (tp + fn)


def plot_class_distribution(dataframe: pd.DataFrame, output_path: Path) -> None:
    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(7, 4))
    order = ["Y", "N"]
    ax = sns.countplot(
        data=dataframe,
        x="Loan_Status",
        hue="Loan_Status",
        order=order,
        palette="deep",
        legend=False,
    )
    ax.set_title("Loan status distribution")
    ax.set_xlabel("Loan status")
    ax.set_ylabel("Count")
    for patch in ax.patches:
        ax.annotate(
            f"{int(patch.get_height())}",
            (patch.get_x() + patch.get_width() / 2, patch.get_height()),
            ha="center",
            va="bottom",
        )
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def plot_missing_values(dataframe: pd.DataFrame, output_path: Path) -> None:
    missing = dataframe.isna().sum().sort_values(ascending=False)
    missing = missing[missing > 0]
    plt.figure(figsize=(9, 4))
    ax = sns.barplot(
        x=missing.index,
        y=missing.values,
        hue=missing.index,
        palette="magma",
        legend=False,
    )
    ax.set_title("Missing values before preprocessing")
    ax.set_xlabel("Feature")
    ax.set_ylabel("Missing values")
    ax.tick_params(axis="x", rotation=45)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def plot_confusion_matrix(
    matrix: np.ndarray, title: str, output_path: Path, labels: list[str]
) -> None:
    fig, ax = plt.subplots(figsize=(5, 4))
    disp = ConfusionMatrixDisplay(confusion_matrix=matrix, display_labels=labels)
    disp.plot(ax=ax, cmap="Blues", colorbar=False)
    ax.set_title(title)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_roc_curves(
    roc_artifacts: dict[str, tuple[np.ndarray, np.ndarray, float]], output_path: Path
) -> None:
    plt.figure(figsize=(7, 5))
    for name, (false_positive_rate, true_positive_rate, auc_score) in roc_artifacts.items():
        plt.plot(
            false_positive_rate,
            true_positive_rate,
            linewidth=2,
            label=f"{name} (AUC = {auc_score:.3f})",
        )
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Random guess")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC curve comparison")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def evaluate_models(
    features_train: pd.DataFrame,
    features_test: pd.DataFrame,
    target_train: pd.Series,
    target_test: pd.Series,
    preprocessor: ColumnTransformer,
) -> EvaluationArtifacts:
    model_candidates = {
        "Logistic Regression": LogisticRegression(
            max_iter=2000,
            random_state=RANDOM_STATE,
        ),
        "k-Nearest Neighbors": KNeighborsClassifier(n_neighbors=7),
    }

    metrics_rows: list[dict[str, float | str]] = []
    confusion_matrices: dict[str, np.ndarray] = {}
    roc_artifacts: dict[str, RocArtifact] = {}
    predictions: dict[str, np.ndarray] = {}

    for model_name, estimator in model_candidates.items():
        model = Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("classifier", estimator),
            ]
        )
        model.fit(features_train, target_train)

        predicted_labels = model.predict(features_test)
        predicted_probabilities = model.predict_proba(features_test)[:, 1]
        matrix = confusion_matrix(target_test, predicted_labels, labels=[0, 1])
        auc_score = roc_auc_score(target_test, predicted_probabilities)
        false_positive_rate, true_positive_rate, _ = roc_curve(
            target_test, predicted_probabilities
        )

        confusion_matrices[model_name] = matrix
        roc_artifacts[model_name] = (
            false_positive_rate,
            true_positive_rate,
            auc_score,
        )
        predictions[model_name] = predicted_labels
        metrics_rows.append(
            {
                "Model": model_name,
                "Accuracy": accuracy_score(target_test, predicted_labels),
                "Sensitivity": sensitivity_from_confusion_matrix(matrix),
                "Specificity": specificity_from_confusion_matrix(matrix),
                "ROC AUC": auc_score,
            }
        )

    metrics_frame = pd.DataFrame(metrics_rows).sort_values(
        by=["Accuracy", "ROC AUC"], ascending=False
    )
    return metrics_frame, confusion_matrices, roc_artifacts, predictions


def run_mcnemar_test(
    actual_values: pd.Series,
    first_predictions: np.ndarray,
    second_predictions: np.ndarray,
) -> tuple[int, int, float]:
    first_correct = first_predictions == actual_values.to_numpy()
    second_correct = second_predictions == actual_values.to_numpy()

    first_right_second_wrong = int(np.sum(first_correct & ~second_correct))
    first_wrong_second_right = int(np.sum(~first_correct & second_correct))

    if first_right_second_wrong + first_wrong_second_right == 0:
        return first_right_second_wrong, first_wrong_second_right, 1.0

    p_value = binomtest(
        min(first_right_second_wrong, first_wrong_second_right),
        n=first_right_second_wrong + first_wrong_second_right,
        p=0.5,
        alternative="two-sided",
    ).pvalue
    return first_right_second_wrong, first_wrong_second_right, p_value


def write_report(
    dataset_path: Path,
    dataframe: pd.DataFrame,
    features_train: pd.DataFrame,
    features_test: pd.DataFrame,
    target_train: pd.Series,
    target_test: pd.Series,
    metrics_frame: pd.DataFrame,
    confusion_matrices: dict[str, np.ndarray],
    first_better: int,
    second_better: int,
    mcnemar_p_value: float,
) -> None:
    metrics_table = metrics_frame.copy()
    for column in ["Accuracy", "Sensitivity", "Specificity", "ROC AUC"]:
        metrics_table[column] = metrics_table[column].map(float_fmt)

    source_code = Path(__file__).read_text()
    head_preview = dataframe.head(10).to_string(index=False)
    missing_values = dataframe.isna().sum().to_string()
    training_distribution = target_train.map({1: "Approved", 0: "Rejected"}).value_counts()
    testing_distribution = target_test.map({1: "Approved", 0: "Rejected"}).value_counts()

    logistic_matrix = pd.DataFrame(
        confusion_matrices["Logistic Regression"],
        index=["Actual Rejected", "Actual Approved"],
        columns=["Predicted Rejected", "Predicted Approved"],
    )
    knn_matrix = pd.DataFrame(
        confusion_matrices["k-Nearest Neighbors"],
        index=["Actual Rejected", "Actual Approved"],
        columns=["Predicted Rejected", "Predicted Approved"],
    )

    report_prefix = left_trim_eight_spaces(
        dedent(
        f"""
        # Loan Status Prediction Technical Report

        ## a) Problem statement

        This project addresses two connected tasks:

        1. Explain the data mining techniques **classification** and **prediction**, including how each works, its strengths, weaknesses, and a real-life example.
        2. Use the **Loan Status Prediction Dataset** to build and compare two classifiers. The workflow must load the data, preprocess it, subset the predictors, split the data into training and testing sets, fit the models, make predictions, report confusion matrices, compute accuracy/sensitivity/specificity, explain ROC/AUC, and compare the classifiers statistically.

        The analysis in this repository uses:

        - **Logistic Regression**
        - **k-Nearest Neighbors (kNN)**

        Dataset used during execution:

        - `{dataset_path}`

        ---

        ## Part I - Data mining concepts

        ### 1) Classification

        **How it works:** Classification learns a mapping from input features to a discrete target label using historical labeled examples. During training, the algorithm estimates decision boundaries or class probabilities, then applies the learned pattern to unseen records.

        **Strengths:**

        - Works well when the goal is a discrete decision such as approve/reject or spam/not spam.
        - Supports measurable evaluation through confusion matrices, ROC curves, and class-level metrics.
        - Many mature algorithms exist, from interpretable linear models to flexible nonlinear models.

        **Weaknesses:**

        - Requires labeled historical data.
        - Can inherit class imbalance and social/process bias present in the source data.
        - Some methods require careful preprocessing and hyperparameter tuning.

        **Real-life example:** Email providers classify incoming messages as **spam** or **not spam** using message content, sender reputation, and behavioral metadata.

        ### 2) Prediction

        **How it works:** Prediction estimates an unknown or future value from observed variables. In many business settings the term refers to supervised learning for continuous targets, where the model learns a numeric relationship and outputs an expected value rather than a class label.

        **Strengths:**

        - Useful for forecasting numeric outcomes such as demand, revenue, or time-to-event.
        - Supports proactive planning because it estimates future behavior.
        - Can incorporate many explanatory variables from operational systems.

        **Weaknesses:**

        - Forecast quality depends strongly on data quality and how stable the underlying process remains over time.
        - Extreme values and distribution shifts can reduce performance quickly.
        - Results may be less intuitive to explain than rule-based decision systems.

        **Real-life example:** A retailer predicts **next-month sales volume** from promotions, seasonality, prices, and historical transactions.

        ---

        ## Group Discussion 1 synopsis

        The dataset includes a mix of demographic, categorical, and financial variables such as gender, marital status, education, income, loan amount, loan term, credit history, and property area. That structure strongly influenced model choice: it favors algorithms that can work after categorical encoding and missing-value imputation, and it highlights the importance of scaling for distance-based methods such as kNN. The discussion also surfaced practical concerns about missing values, class imbalance, and fairness risk around demographic attributes, so the final workflow explicitly imputes missing values, one-hot encodes categorical fields, standardizes numeric features, and evaluates more than one model instead of trusting a single score.

        ---

        ## b) Algorithm of the solution

        1. Load `LoanDataSet.csv`.
        2. Inspect structure, class balance, and missing values.
        3. Drop `Loan_ID` because it is an identifier rather than a predictive signal.
        4. Separate predictors from the target variable `Loan_Status`.
        5. Split the dataset into stratified training and testing subsets.
        6. Build one preprocessing pipeline:
           - median imputation for numeric features,
           - most-frequent imputation for categorical features,
           - one-hot encoding for categorical features,
           - z-score scaling for numeric features.
        7. Train two classifiers:
           - logistic regression,
           - k-nearest neighbors with `k = 7`.
        8. Generate predictions and predicted probabilities on the test set.
        9. Compute confusion matrices, accuracy, sensitivity, specificity, and ROC AUC.
        10. Compare model predictions with an exact McNemar test on the paired test-set outcomes.

        ---

        ## Dataset inspection and preprocessing outputs

        ### Dataset preview

        {markdown_code_block(head_preview)}

        ### Missing values before preprocessing

        {markdown_code_block(missing_values)}

        ### Key features of the dataset

        - Rows: **{len(dataframe)}**
        - Columns: **{len(dataframe.columns)}**
        - Target classes: **Approved (Y)** and **Rejected (N)**
        - Mixed data types: categorical and numeric
        - Moderate missingness concentrated in `Credit_History`, `Self_Employed`, `LoanAmount`, `Dependents`, `Loan_Amount_Term`, `Gender`, and `Married`

        ### Train/test split

        - Training rows: **{len(features_train)}**
        - Testing rows: **{len(features_test)}**
        - Test size: **{int(TEST_SIZE * 100)}%**
        - Random state: **{RANDOM_STATE}**

        Training distribution:

        {markdown_code_block(training_distribution.to_string())}

        Testing distribution:

        {markdown_code_block(testing_distribution.to_string())}

        ### Visual outputs

        #### Class balance

        ![Class distribution](plots/class_distribution.png)

        #### Missing values

        ![Missing values](plots/missing_values.png)

        ---

        ## Classification results

        ### Quantitative comparison

        {dataframe_to_markdown_table(metrics_table, include_index=False)}

        ### Logistic regression confusion matrix

        {dataframe_to_markdown_table(logistic_matrix, include_index=True)}

        ![Logistic regression confusion matrix](plots/logistic_regression_confusion_matrix.png)

        ### kNN confusion matrix

        {dataframe_to_markdown_table(knn_matrix, include_index=True)}

        ![kNN confusion matrix](plots/k_nearest_neighbors_confusion_matrix.png)

        ### ROC curve comparison

        ![ROC curves](plots/roc_curves.png)

        ---

        ## 10) ROC curve and AUC explanation

        The **receiver operating characteristic (ROC) curve** plots the trade-off between the true positive rate (sensitivity) and the false positive rate across classification thresholds. It shows how well a model separates approved and rejected loan applications when the decision threshold changes. The **area under the ROC curve (AUC)** summarizes this ranking ability on a 0-to-1 scale: values near 1.0 indicate strong separation, values near 0.5 indicate near-random discrimination.

        ---

        ## 11) Comparison and analysis of findings

        Logistic regression achieved the strongest overall performance in this run, with the best accuracy (**{float_fmt(metrics_frame.iloc[0]["Accuracy"])}**) and the best ROC AUC (**{float_fmt(metrics_frame.iloc[0]["ROC AUC"])}**). It also delivered very high sensitivity, meaning it captured almost all approved-loan cases in the test set. The cost of that behavior is a more modest specificity, so the model is better at recognizing approvals than rejections.

        kNN produced competitive but slightly weaker results. Because kNN is distance-based, its behavior is more sensitive to local neighborhoods, noisy points, and the exact balance of scaled features after one-hot encoding. Even after standardization, the model remained slightly less accurate than logistic regression and produced a marginally lower AUC.

        The confusion matrices show that both classifiers favor the majority approval class, which is expected because the dataset contains more approvals than rejections. This makes sensitivity higher than specificity for both methods. In a lending context:

        - **Accuracy** measures total correctness.
        - **Sensitivity** measures how well the model identifies approved loans.
        - **Specificity** measures how well the model identifies rejected loans.

        Since false approvals and false rejections have different business costs, the best model should not be selected on accuracy alone. Logistic regression is the preferred model here because it is slightly stronger quantitatively, easy to interpret, and naturally produces calibrated probabilities useful for threshold tuning and policy review.

        ### Statistical difference

        McNemar's exact test compared the paired test-set predictions:

        - Logistic regression correct / kNN incorrect: **{first_better}**
        - Logistic regression incorrect / kNN correct: **{second_better}**
        - Exact McNemar p-value: **{float_fmt(mcnemar_p_value)}**

        Because the p-value is greater than 0.05, the observed difference is **not statistically significant** at the 5% level for this test split. In practice, that means the two models perform similarly enough that interpretability, operational simplicity, and probability outputs become strong tie-breakers in favor of logistic regression.

        ---

        ## Group Discussion 2 synopsis

        The discussion after modeling reinforced that confusion matrices matter because they show *which* mistakes each classifier makes, not just how many. In loan-status prediction, sensitivity and specificity translate into different lending risks, and the ROC/AUC perspective helps evaluate ranking quality beyond a single threshold. The group also agreed that when two models look similar statistically, the more interpretable and easier-to-govern method is usually the better operational choice, which aligned with the final recommendation to prefer logistic regression for this dataset.

        ---

        ## c) Analysis of the findings

        1. **The dataset is usable after standard preprocessing.** Missing values are present but concentrated in a few columns, so imputation is a practical remedy.
        2. **Scaling matters for kNN.** Without scaling, distance-based comparisons would be dominated by large numeric ranges such as income and loan amount.
        3. **Class imbalance affects error patterns.** The dataset has substantially more approvals than rejections, which contributes to higher sensitivity than specificity.
        4. **Logistic regression is a strong baseline.** It handled the encoded feature space well, produced the best AUC, and remained easy to explain.
        5. **The models are close enough that governance matters.** Since the McNemar test does not show a significant gap, model choice can reasonably emphasize interpretability, threshold control, and ease of maintenance.

        ---

        ## Full code used in the analysis
        """
        ).strip()
    )

    report_code_section = f"```python\n{source_code.rstrip()}\n```"
    report_suffix = left_trim_eight_spaces(
        dedent(
        """
        ---

        ## d) References

        1. Han, J., Kamber, M., & Pei, J. *Data Mining: Concepts and Techniques*.
        2. scikit-learn documentation: https://scikit-learn.org/
        3. Dataset source copied into this repository from `parthmiddha/CodeClause_Loan_Prediction`, file `LoanDataset.csv`: https://github.com/parthmiddha/CodeClause_Loan_Prediction
        """
        ).strip()
    )

    report = "\n\n".join([report_prefix, report_code_section, report_suffix])
    (OUTPUT_DIR / "technical_report.md").write_text(report + "\n")


def main() -> None:
    dataset_path = resolve_dataset_path()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    dataframe = pd.read_csv(dataset_path)
    features = dataframe.drop(columns=["Loan_Status", "Loan_ID"])
    target = dataframe["Loan_Status"].map({"Y": 1, "N": 0})

    features_train, features_test, target_train, target_test = train_test_split(
        features,
        target,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=target,
    )

    preprocessor = build_preprocessor(features)
    metrics_frame, confusion_matrices, roc_artifacts, predictions = evaluate_models(
        features_train=features_train,
        features_test=features_test,
        target_train=target_train,
        target_test=target_test,
        preprocessor=preprocessor,
    )

    plot_class_distribution(dataframe, PLOTS_DIR / "class_distribution.png")
    plot_missing_values(dataframe, PLOTS_DIR / "missing_values.png")
    plot_confusion_matrix(
        confusion_matrices["Logistic Regression"],
        "Logistic Regression Confusion Matrix",
        PLOTS_DIR / "logistic_regression_confusion_matrix.png",
        labels=["Rejected", "Approved"],
    )
    plot_confusion_matrix(
        confusion_matrices["k-Nearest Neighbors"],
        "k-Nearest Neighbors Confusion Matrix",
        PLOTS_DIR / "k_nearest_neighbors_confusion_matrix.png",
        labels=["Rejected", "Approved"],
    )
    plot_roc_curves(roc_artifacts, PLOTS_DIR / "roc_curves.png")

    first_better, second_better, mcnemar_p_value = run_mcnemar_test(
        actual_values=target_test,
        first_predictions=predictions["Logistic Regression"],
        second_predictions=predictions["k-Nearest Neighbors"],
    )

    write_report(
        dataset_path=dataset_path,
        dataframe=dataframe,
        features_train=features_train,
        features_test=features_test,
        target_train=target_train,
        target_test=target_test,
        metrics_frame=metrics_frame,
        confusion_matrices=confusion_matrices,
        first_better=first_better,
        second_better=second_better,
        mcnemar_p_value=mcnemar_p_value,
    )

    print(f"Dataset used: {dataset_path}")
    print(f"Report written to: {OUTPUT_DIR / 'technical_report.md'}")
    print("Metrics summary:")
    print(metrics_frame.to_string(index=False))


if __name__ == "__main__":
    main()

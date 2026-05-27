# BenchmarkDataMiningClassification

This repository contains a reproducible loan-status classification workflow and the generated technical report required by the assignment.

## Files

- `/tmp/workspace/Faus7679/BenchmarkDataMiningClassification/analysis/loan_status_analysis.py` - end-to-end analysis script
- `/tmp/workspace/Faus7679/BenchmarkDataMiningClassification/data/LoanDataSet.csv` - local dataset copy for offline execution
- `/tmp/workspace/Faus7679/BenchmarkDataMiningClassification/docs/technical_report.md` - comprehensive markdown report
- `/tmp/workspace/Faus7679/BenchmarkDataMiningClassification/docs/plots/` - generated plots used in the report
- `/tmp/workspace/Faus7679/BenchmarkDataMiningClassification/requirements.txt` - Python dependencies

## How to run the project

1. Create or activate a Python 3.12 environment.
2. Install dependencies:

   ```bash
   python -m pip install -r /tmp/workspace/Faus7679/BenchmarkDataMiningClassification/requirements.txt
   ```

3. Put `LoanDataSet.csv` in `~/Downloads/LoanDataSet.csv` if you want to use your own download.
   - If that file is not present, the script automatically falls back to `/tmp/workspace/Faus7679/BenchmarkDataMiningClassification/data/LoanDataSet.csv`.
4. Run the analysis:

   ```bash
   python /tmp/workspace/Faus7679/BenchmarkDataMiningClassification/analysis/loan_status_analysis.py
   ```

## What the script does

1. Loads the loan dataset.
2. Inspects the structure, target balance, and missing values.
3. Drops `Loan_ID` and keeps the remaining predictive features.
4. Splits the data into stratified training and testing sets.
5. Preprocesses the data with:
   - median imputation for numeric values,
   - most-frequent imputation for categorical values,
   - one-hot encoding for categorical variables,
   - standard scaling for numeric variables.
6. Trains two classifiers:
   - logistic regression,
   - k-nearest neighbors.
7. Generates predictions, confusion matrices, ROC curves, accuracy, sensitivity, specificity, and ROC AUC.
8. Runs an exact McNemar comparison between the two models.
9. Writes the markdown report and plot images into `/tmp/workspace/Faus7679/BenchmarkDataMiningClassification/docs/`.

## Assignment coverage

- Part I explanations for classification and prediction are included in the report.
- Part II preprocessing, modeling, visualizations, confusion matrices, metrics, ROC/AUC explanation, and model comparison are included in the report.
- The README documents each step of the solution workflow.

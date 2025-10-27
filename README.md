# Detection of Disengagement from Voluntary Quizzes  
### An Explainable Machine Learning Approach in Higher Distance Education

**Behnam Parsaeifard**, **Christof Imhof**, **Tansu Pancar**, **Ioan-Sorin Comsa**, **Martin Hlosta**, **Nicole Bergamin**, and **Per Bergamin**

---

## Overview

This repository contains the implementation accompanying the preprint:

> **Detection of Disengagement from Voluntary Quizzes: An Explainable Machine Learning Approach in Higher Distance Education**  
> [arXiv:2507.02681](https://arxiv.org/pdf/2507.02681?)

The project is part of the **AIDA** framework —  
**A**nalysis and **I**ntervention in the context of **D**ropout **A**lerts —  
and focuses on predicting voluntary quiz disengagement in higher distance education using interpretable machine learning methods.

---

## Project Layout

```
.
├── .gitignore                # excludes raw data and environment files
├── .env                      # local configuration (not tracked)
├── requirements.txt           # Python dependencies
├── data/
│   ├── raw/                  # input files (not tracked)
│   └── preprocessed/         # generated artifacts (.csv, .npy)
├── models/                   # trained models (.pkl)
├── notebooks/                # optional analysis notebooks
└── src/
    ├── data_loader.py        # load and merge Moodle logs and quiz attempts
    ├── data_preprocessing.py # feature extraction, labeling, data split
    ├── train.py              # model training and hyperparameter tuning
    ├── evaluate.py           # evaluation and LaTeX table export
    └── __init__.py
```

---

## Environment Setup

### 1. Create and Activate a Conda Environment

```powershell
conda create -n aida python=3.11 -y
conda activate aida
```

### 2. Install Dependencies

```powershell
pip install -r requirements.txt
```

### 3. Define Project Root

Create a `.env` file in the repository root:

```
ROOT=PATH/TO/PROJECT
```

Scripts read this variable to locate input and output directories.  
If missing, the default is the current working directory.

---

## Data Preparation

`data/raw/` contains the required quiz and log files.
(these files are not tracked by Git for privacy and size reasons).

Expected input structure:

```
data/raw/
├── Quiz attempts.xlsx
├── log attempts.
```

---

## Pipeline

### 1. Load and Merge Raw Data

```powershell
python .\src\data_loader.py
```

**Purpose:**  
Reads quiz attempts and log files, filters demo/introductory courses and non-student users, merges logs with quiz data, and estimates quiz and course time windows.

**Outputs:**
- `data/preprocessed/clean_merged_data.csv`

**Main Functions:**
- `load_quiz(path)` — reads quiz attempt data, filters by course/user, extracts semester, and removes invalid preknowledge tests.  
- `load_log(paths)` — concatenates and cleans Moodle logs.  
- `merge_tables(df_quiz, df_log)` — merges logs with quiz attempts and adds derived time features.

---

### 2. Feature Engineering and Labeling

```powershell
python .\src\data_preprocessing.py
```

**Purpose:**  
Transforms merged event data into per-attempt, per-day features and assigns binary labels indicating quiz submission.

**Outputs:**
- `data/preprocessed/train-data.npy` (features | labels | groups)
- `data/preprocessed/test-data.npy` (features | labels)

**Main Steps:**
- Derives relative days from quiz start and submission.
- Encodes temporal activity (weekday/weekend × morning/afternoon/evening).
- Calculates inactivity spans and cumulative activity counts.
- Computes temporal statistics (min, max, mean, median, SD, skewness, kurtosis).
- Labels each day as submitted (`did_submit = True`) or not.
- Splits by semester.

**Key Functions:**
- `labeling_3(df)` — generates labels for the first 20 days.  
- `split_data(df, train_semesters, test_semesters)` — separates data chronologically.  
- `preprocess(df_merged)` — full feature generation pipeline.  
- `save_data` / `load_data` — save and reload preprocessed arrays.  
- `plot_feature_correlation(...)` — optional visualization of correlations.

---

### 3. Model Training

```powershell
python .\src\train.py --cv 4 --n_jobs -1 --verbose 5 --scoring roc_auc
```

**Purpose:**  
Trains and tunes baseline classifiers using grouped cross-validation and saves best models.

**Outputs:**
```
models/{model_name}.pkl
```

**Details:**
- Each model is wrapped in a `Pipeline(StandardScaler(), clf)` to ensure scaling within CV folds.
- Uses `StratifiedGroupKFold` to prevent leakage between attempts.
- Default grid searches include Decision Tree and Logistic Regression.
- Additional models (Random Forest, MLP, KNN, Gradient Boosting, XGBoost, SVC, Naive Bayes) can be enabled by uncommenting them in `get_param_grids()`.

**Key Functions:**
- `get_param_grids()` — defines model hyperparameter grids.  
- `save_model(model, name)` — saves trained models.  
- `report_performance(...)` — prints train/test reports and balanced accuracy.

A simple **VotingClassifier** combining the best estimators is also trained and saved.

---

### 4. Evaluation

```powershell
python .\src\evaluate.py
```

**Purpose:**  
Evaluates all trained models on the test set and prints a LaTeX-formatted summary table.

**Metrics:**
- PPV / NPV  
- TPR / FPR / TNR / FNR  
- F1-scores per class and weighted average  
- Balanced Accuracy  
- ROC AUC

**Outputs:**
- Console output with metrics
- LaTeX table ready to paste into the paper

**Key Functions:**
- `calculate_metrics_binary_classes(...)` — computes detailed binary metrics.  
- `load_model(name)` — loads pickled models from disk.  
- `ev_to_latex(ev, ...)` — converts evaluation results to LaTeX format.

---

## Configuration

You may adjust key settings:

- **Semester split** in `data_preprocessing.py`:
  ```python
  train_semesters = ['HS17/18', 'FS18', 'HS18/19']
  test_semesters  = ['FS19']
  ```

- **Feature selection** (`feature_names_all` and `excluded_features` at the top of `data_preprocessing.py`).

- **Event filtering**:
  ```python
  events_included = [
      'attempt_started', 'attempt_viewed',
      'attempt_summary_viewed', 'course_module_viewed'
  ]
  ```

- **Hyperparameter grids** in `train.py` → `get_param_grids()`.

---

## Explainability Analysis

The notebook `explainability.ipynb` contains the explainability and model interpretation.
It demonstrates how model predictions are analyzed and visualized using SHAP (SHapley Additive Explanations) to identify which learning behavior features most influence the predicted dropout or disengagement risk.

Key steps included:

* Loading trained models and datasets from `data/preprocessed/` and `models/`
* Sampling background data from the training set to serve as a baseline for SHAP
* Computing SHAP values for test samples (both batched and parallel options available)
* Visualizing results through summary plots, feature impact rankings, and distribution analyses

This notebook provides interpretable insights into how individual features—such as inactivity duration, quiz attempts, or previous performance—affect model outputs.
It supports transparent decision-making and educational feedback, ensuring the AIDA models remain explainable and actionable in real-world learning analytics contexts.

---

## Reproducibility

- Cross-validation is **StratifiedGroupKFold** (grouped by attempt ID).  
- Standardization occurs inside the CV pipeline, preventing leakage.  
- Random states can be fixed in model definitions if deterministic runs are needed.

---

## Troubleshooting

- **XGBoost model loading:**  
  Ensure the same XGBoost version is installed or use `save_model()` / `load_model()` (JSON format).

- **`predict_proba` not found:**  
  For SVC, set `probability=True` when enabling it in `get_param_grids()`.

- **Environment export too large:**  
  Use `requirements.txt` instead of `conda env export` to keep the environment lightweight.

---

## License and Citation

If you use this repository, please cite the corresponding paper:

> Parsaeifard, B., Imhof, C., Pancar, T., Comsa, I.-S., Hlosta, M., Bergamin, N., & Bergamin, P.  
> *Detection of Disengagement from Voluntary Quizzes: An Explainable Machine Learning Approach in Higher Distance Education.*  
> arXiv preprint arXiv:2507.02681, 2025.

---

## Notes for Contributors

- Do **not** commit contents of `data/raw/` or `.env`.  
- Keep docstrings consistent with the paper terminology.  
- Contributions improving reproducibility, documentation, or testing are welcome.

---

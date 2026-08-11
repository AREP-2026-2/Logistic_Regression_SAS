# Heart Disease Risk Prediction — Logistic Regression

## Exercise Summary

This project implements **logistic regression from scratch** (NumPy/Pandas/
Matplotlib only — no scikit-learn for the core model) to predict the
presence of heart disease from clinical features. The work covers:

- Exploratory data analysis and preprocessing of the Heart Disease dataset.
- A from-scratch logistic regression model: sigmoid function, binary
  cross-entropy cost, gradients, and gradient descent, trained on six
  clinical features.
- Visualization of decision boundaries for pairs of clinically relevant
  features (age–cholesterol, blood pressure–max heart rate, ST
  depression–number of vessels).
- L2-regularized logistic regression, with a sweep over
  `lambda in [0, 0.001, 0.01, 0.1, 1]` and a comparison of regularized vs.
  unregularized decision boundaries.
- Training and testing the same model in **Amazon SageMaker** (AWS
  Academy), using the exact same preprocessed data as the local run, for a
  direct comparison. No SageMaker endpoint or deployment service was created
  or used, per the AWS Academy account limitations for this course.

All implementation, plots, metrics tables and discussion live in
[`heart_disease_lr_analysis.ipynb`](heart_disease_lr_analysis.ipynb).

## Dataset Description

**Source:** [Heart Disease Dataset — Kaggle (neurocipher)](https://www.kaggle.com/datasets/neurocipher/heartdisease)

The dataset is a widely used republication of the UCI Cleveland Heart
Disease data: clinical records with 13 features (age, sex, chest pain type,
resting blood pressure, cholesterol, fasting blood sugar, resting ECG,
maximum heart rate, exercise-induced angina, ST depression, slope of the
peak exercise ST segment, number of major vessels, and thalassemia) plus a
binary target (`1` = disease present, `0` = absent).

The raw CSV (`heart.csv`, 1025 rows) contains many exact duplicate rows —
after removing duplicates, the underlying population is ~300 unique
patients, matching the ~303 patients described in the original UCI dataset.
A small number of rows also contain documented invalid categorical codes
(`ca = 4`, `thal = 0`) which were removed during preprocessing. See Step 1 of
the notebook for the full data-cleaning walkthrough.

Six features were selected for modeling: **age, resting blood pressure
(`trestbps`), cholesterol (`chol`), maximum heart rate (`thalach`), ST
depression (`oldpeak`) and number of major vessels (`ca`)**.

## SageMaker Evidence

*(To be filled in after completing Step 5 in the AWS Academy console — see
the "Step 5" section of `heart_disease_lr_analysis.ipynb` for the exact
notebook cells to run there.)*

**Environment:** *(fill in — e.g. SageMaker notebook instance type, kernel,
AWS region)*

1. **Notebook instance running / InService**
   `![SageMaker notebook instance](images/sagemaker_instance.png)`
2. **Training execution completed** (cost printed at the end of training)
   `![SageMaker training completed](images/sagemaker_training.png)`
3. **Test-set metrics**
   `![SageMaker test metrics](images/sagemaker_metrics.png)`

**Test-set results (SageMaker):**

| Metric | Value |
|---|---|
| Accuracy | *(fill in)* |
| Precision | *(fill in)* |
| Recall | *(fill in)* |
| F1 | *(fill in)* |

**Comparison with local execution:** *(fill in — both runs use the same
`heart_train.csv` / `heart_test.csv` and hyperparameters, so results should
be very close; note and explain any difference actually observed.)*

## Repository Contents

- `heart_disease_lr_analysis.ipynb` — end-to-end notebook (EDA, model,
  visualization, regularization, SageMaker write-up, final insights).
- `heart.csv` — raw dataset downloaded from Kaggle.
- `heart_train.csv`, `heart_test.csv` — preprocessed train/test split
  exported by the notebook, used to reproduce results in SageMaker.
- `README.md` — this file.
- `images/` — SageMaker evidence screenshots.

## How to Run Locally

```bash
pip install numpy pandas matplotlib
jupyter notebook heart_disease_lr_analysis.ipynb
```

Run all cells top to bottom. Step 5's code cells are meant to be copied into
a SageMaker notebook, not executed locally (they are included here for
reference and reproducibility).

# SPH6004 Individual Assignment

Predicting ICU discharge (survival) using MIMIC-IV v3.1 data.

## Files

| File | Description |
|------|-------------|
| `feature_selection.py` | Feature selection pipeline |
| `predictive_models.py` | Model training and evaluation |
| `generate_plots.py` | Plot generation |
| `*.png` | Figures used in the report |

## Feature Selection

Six methods were applied and features were kept if selected by at least 4 of them:
variance threshold, correlation filter (>0.90), ANOVA F-test, L1 regularization, random forest importance, and RFE. This reduced the feature set from 140 to 36.

## Models

Seven classifiers from the first half of the course: Logistic Regression, LDA, QDA, KNN, Gaussian Naive Bayes, Decision Tree, and Linear SVM.

Best model: **Logistic Regression** (AUC-ROC = 0.8876).

## Requirements

```
Python 3.11+
pandas, numpy, scikit-learn, matplotlib, seaborn
```

## Dataset

MIMIC-IV v3.1 — Johnson et al., *Sci Data* 9, 3 (2023).

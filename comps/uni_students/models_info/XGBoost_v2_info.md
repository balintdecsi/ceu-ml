# XGBoost (v2 — Improved Pipeline)

## Preprocessing
- Smart missing value handling (structural vs behavioral)
- Engineered features: engagement trends, access span, submission behavior, etc.
- One-hot encoding (fit on train only)
- Scaling: No (tree model)

## Hyperparameters
- n_estimators: 500
- max_depth: 8
- learning_rate: 0.0299058938693663
- subsample: 0.8285586096034029
- colsample_bytree: 0.7777243706586128
- min_child_weight: 1
- reg_alpha: 0.1767218232266506
- reg_lambda: 0.0042733023193754
- objective: multi:softprob
- num_class: 3
- random_state: 42
- n_jobs: -1
- verbosity: 0

## Cross-Validation Performance (5-fold Stratified)
- **Mean Accuracy**: 0.6511 ± 0.0068
- **Per-fold**: 0.6607, 0.6500, 0.6529, 0.6396, 0.6524

## Full-Train Classification Report (on last fold val set)
```text
              precision    recall  f1-score   support

        Fail       0.40      0.39      0.40      1057
        Pass       0.71      0.81      0.76      2307
   Withdrawn       0.73      0.60      0.66      1524

    accuracy                           0.65      4888
   macro avg       0.62      0.60      0.60      4888
weighted avg       0.65      0.65      0.65      4888

```

## Top 15 Feature Importances
- chose_not_to_submit: 0.2217
- stopped_early: 0.1887
- min_score_missing: 0.0775
- code_module_GGG: 0.0155
- no_assess_available: 0.0151
- code_module_EEE: 0.0129
- mean_score_missing: 0.0124
- num_assessments: 0.0118
- last_access_day: 0.0116
- mean_score_TMA_missing: 0.0103
- min_score: 0.0098
- code_module_AAA: 0.0080
- mean_score_CMA: 0.0074
- max_score: 0.0074
- highest_education_Lower_Than_A_Level: 0.0073
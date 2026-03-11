# XGBoost (v3 — Accuracy-Optimized)

## Preprocessing
- Smart missing value handling (structural vs behavioral)
- Engineered features: engagement trends, access span, submission behavior
- One-hot encoding (fit on train only)
- Scaling: No (tree model)
- **No class balancing** (accuracy metric rewards majority class)

## Hyperparameters
- n_estimators: 300
- max_depth: 6
- learning_rate: 0.020644253618384636
- subsample: 0.7740218032969832
- colsample_bytree: 0.6857562925335298
- min_child_weight: 5
- reg_alpha: 0.03909195674447074
- reg_lambda: 0.1957210208562269
- objective: multi:softprob
- num_class: 3
- random_state: 42
- n_jobs: -1
- verbosity: 0

## Cross-Validation Performance (5-fold Stratified)
- **Mean Accuracy**: 0.6660 ± 0.0032
- **Per-fold**: 0.6723, 0.6646, 0.6650, 0.6633, 0.6647

## Classification Report (last fold val set)
```text
              precision    recall  f1-score   support

        Fail       0.44      0.19      0.26      1057
        Pass       0.67      0.91      0.77      2307
   Withdrawn       0.74      0.62      0.68      1524

    accuracy                           0.66      4888
   macro avg       0.62      0.57      0.57      4888
weighted avg       0.64      0.66      0.63      4888

```

## Top 15 Feature Importances
- chose_not_to_submit: 0.2462
- stopped_early: 0.1290
- min_score_missing: 0.0432
- mean_score_missing: 0.0285
- no_assess_available: 0.0212
- last_access_day: 0.0211
- num_assessments: 0.0195
- min_score: 0.0165
- mean_score_TMA_missing: 0.0164
- code_module_GGG: 0.0151
- code_module_EEE: 0.0137
- late_engagement: 0.0129
- max_score: 0.0127
- mean_score: 0.0118
- max_score_missing: 0.0115
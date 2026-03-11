# Tuned XGBoost

## Validation Strategy
- 3-fold stratified cross-validation
- Bootstrap confidence interval computed from the fold-level accuracies

## Best Hyperparameters
- colsample_bytree: 0.8
- learning_rate: 0.05
- max_depth: 4
- min_child_weight: 3
- n_estimators: 250
- subsample: 0.8

## Cross-Validation Summary
- Fold accuracies: 0.6684, 0.6649, 0.6638
- Mean accuracy: 0.6657
- Std. accuracy: 0.0024
- Bootstrap 95% CI: [0.6638, 0.6684]

## Top Feature Importances
- chose_not_to_submit: 0.2794
- behavioral_score_missing: 0.2272
- min_score: 0.0245
- mean_score: 0.0207
- last_access_day: 0.0203
- code_module_EEE: 0.0151
- has_no_assessments_available: 0.0149
- highest_education_Lower_Than_A_Level: 0.0131
- num_assessments: 0.0118
- submitted_any_assessment: 0.0117

## Out-of-Fold Classification Report
```text
              precision    recall  f1-score   support

        Fail       0.45      0.21      0.29      5289
        Pass       0.67      0.91      0.77     11538
   Withdrawn       0.74      0.61      0.67      7617

    accuracy                           0.67     24444
   macro avg       0.62      0.58      0.58     24444
weighted avg       0.64      0.67      0.64     24444

```
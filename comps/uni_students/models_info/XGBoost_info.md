# XGBoost

## Hyperparameters
- n_estimators: 100
- max_depth: 6
- learning_rate: 0.1
- objective: 'multi:softmax'

## Validation Performance
- **Accuracy**: 0.6658

## Top 10 Feature Importances
- last_access_day: 0.1224
- code_module_GGG: 0.0765
- num_assessments: 0.0567
- max_score: 0.0480
- mean_score_TMA_is_missing: 0.0425
- code_module_CCC: 0.0310
- min_score: 0.0276
- max_days_late: 0.0258
- code_module_FFF: 0.0203
- code_module_EEE: 0.0179

## Classification Report
```text
              precision    recall  f1-score   support

        Fail       0.44      0.21      0.29      1055
        Pass       0.68      0.90      0.78      2344
   Withdrawn       0.72      0.62      0.66      1490

    accuracy                           0.67      4889
   macro avg       0.61      0.58      0.58      4889
weighted avg       0.64      0.67      0.64      4889

```

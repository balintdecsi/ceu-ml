# Ensemble_SoftVoting (v3 — Accuracy-Optimized)

## Preprocessing
- Smart missing value handling (structural vs behavioral)
- Engineered features: engagement trends, access span, submission behavior
- One-hot encoding (fit on train only)
- Scaling: No (tree model)
- **No class balancing** (accuracy metric rewards majority class)

## Hyperparameters
- models: XGBoost + LightGBM + RandomForest
- method: Average predicted probabilities
- class_balancing: None (accuracy-optimized)

## Cross-Validation Performance (5-fold Stratified)
- **Mean Accuracy**: 0.6646 ± 0.0044
- **Per-fold**: 0.6719, 0.6660, 0.6648, 0.6592, 0.6612

## Classification Report (last fold val set)
```text
              precision    recall  f1-score   support

        Fail       0.44      0.21      0.29      1057
        Pass       0.67      0.89      0.77      2307
   Withdrawn       0.73      0.62      0.67      1524

    accuracy                           0.66      4888
   macro avg       0.61      0.58      0.57      4888
weighted avg       0.64      0.66      0.63      4888

```

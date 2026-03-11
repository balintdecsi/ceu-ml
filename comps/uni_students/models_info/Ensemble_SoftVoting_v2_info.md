# Ensemble_SoftVoting (v2 — Improved Pipeline)

## Preprocessing
- Smart missing value handling (structural vs behavioral)
- Engineered features: engagement trends, access span, submission behavior, etc.
- One-hot encoding (fit on train only)
- Scaling: No (tree model)

## Hyperparameters
- models: XGBoost + LightGBM + RandomForest
- method: Average predicted probabilities
- class_balancing: sample_weight (XGB/LGBM) + class_weight (RF)

## Cross-Validation Performance (5-fold Stratified)
- **Mean Accuracy**: 0.6541 ± 0.0052
- **Per-fold**: 0.6590, 0.6545, 0.6605, 0.6466, 0.6502

## Full-Train Classification Report (on last fold val set)
```text
              precision    recall  f1-score   support

        Fail       0.40      0.37      0.38      1057
        Pass       0.70      0.81      0.75      2307
   Withdrawn       0.73      0.60      0.66      1524

    accuracy                           0.65      4888
   macro avg       0.61      0.59      0.60      4888
weighted avg       0.65      0.65      0.64      4888

```

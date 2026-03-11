# LogisticRegression (v3 — Accuracy-Optimized)

## Preprocessing
- Smart missing value handling (structural vs behavioral)
- Engineered features: engagement trends, access span, submission behavior
- One-hot encoding (fit on train only)
- Scaling: Yes
- **No class balancing** (accuracy metric rewards majority class)

## Hyperparameters
- solver: lbfgs
- max_iter: 2000
- C: 0.0859

## Cross-Validation Performance (5-fold Stratified)
- **Mean Accuracy**: 0.6612 ± 0.0020
- **Per-fold**: 0.6625, 0.6580, 0.6631, 0.6596, 0.6626

## Classification Report (last fold val set)
```text
              precision    recall  f1-score   support

        Fail       0.45      0.19      0.27      1057
        Pass       0.67      0.90      0.77      2307
   Withdrawn       0.73      0.63      0.68      1524

    accuracy                           0.66      4888
   macro avg       0.61      0.57      0.57      4888
weighted avg       0.64      0.66      0.63      4888

```

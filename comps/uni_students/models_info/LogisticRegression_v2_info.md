# LogisticRegression (v2 — Improved Pipeline)

## Preprocessing
- Smart missing value handling (structural vs behavioral)
- Engineered features: engagement trends, access span, submission behavior, etc.
- One-hot encoding (fit on train only)
- Scaling: Yes

## Hyperparameters
- solver: lbfgs
- max_iter: 2000
- class_weight: balanced
- C: 0.5

## Cross-Validation Performance (5-fold Stratified)
- **Mean Accuracy**: 0.6325 ± 0.0070
- **Per-fold**: 0.6359, 0.6423, 0.6310, 0.6210, 0.6324

## Full-Train Classification Report (on last fold val set)
```text
              precision    recall  f1-score   support

        Fail       0.36      0.46      0.41      1057
        Pass       0.73      0.73      0.73      2307
   Withdrawn       0.74      0.61      0.66      1524

    accuracy                           0.63      4888
   macro avg       0.61      0.60      0.60      4888
weighted avg       0.65      0.63      0.64      4888

```

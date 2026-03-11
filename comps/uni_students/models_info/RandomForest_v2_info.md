# RandomForest (v2 — Improved Pipeline)

## Preprocessing
- Smart missing value handling (structural vs behavioral)
- Engineered features: engagement trends, access span, submission behavior, etc.
- One-hot encoding (fit on train only)
- Scaling: No (tree model)

## Hyperparameters
- n_estimators: 300
- max_depth: 23
- min_samples_split: 4
- min_samples_leaf: 4
- max_features: sqrt
- class_weight: balanced
- random_state: 42
- n_jobs: -1

## Cross-Validation Performance (5-fold Stratified)
- **Mean Accuracy**: 0.6557 ± 0.0042
- **Per-fold**: 0.6619, 0.6578, 0.6539, 0.6492, 0.6559

## Full-Train Classification Report (on last fold val set)
```text
              precision    recall  f1-score   support

        Fail       0.41      0.32      0.36      1057
        Pass       0.69      0.85      0.76      2307
   Withdrawn       0.75      0.60      0.67      1524

    accuracy                           0.66      4888
   macro avg       0.61      0.59      0.59      4888
weighted avg       0.65      0.66      0.64      4888

```

## Top 15 Feature Importances
- chose_not_to_submit: 0.0422
- late_engagement: 0.0314
- last_access_day: 0.0310
- days_before_course_start: 0.0273
- click_weekly_mean: 0.0272
- total_clicks: 0.0271
- num_active_days: 0.0268
- click_ratio_w4_w1: 0.0263
- active_day_ratio: 0.0256
- std_clicks_per_day: 0.0246
- clicks_homepage: 0.0239
- mean_clicks_per_day: 0.0238
- click_weekly_std: 0.0234
- clicks_week4: 0.0229
- vle_access_span: 0.0224
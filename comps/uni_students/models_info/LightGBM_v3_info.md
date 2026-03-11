# LightGBM (v3 — Accuracy-Optimized)

## Preprocessing
- Smart missing value handling (structural vs behavioral)
- Engineered features: engagement trends, access span, submission behavior
- One-hot encoding (fit on train only)
- Scaling: No (tree model)
- **No class balancing** (accuracy metric rewards majority class)

## Hyperparameters
- n_estimators: 600
- max_depth: 6
- learning_rate: 0.022732512887240954
- subsample: 0.7777455804289155
- colsample_bytree: 0.6975549966080241
- min_child_samples: 32
- reg_alpha: 0.22819491501562741
- reg_lambda: 1.9132570498012023
- num_leaves: 63
- objective: multiclass
- num_class: 3
- random_state: 42
- n_jobs: -1
- verbosity: -1

## Cross-Validation Performance (5-fold Stratified)
- **Mean Accuracy**: 0.6648 ± 0.0058
- **Per-fold**: 0.6748, 0.6646, 0.6646, 0.6568, 0.6635

## Classification Report (last fold val set)
```text
              precision    recall  f1-score   support

        Fail       0.43      0.23      0.30      1057
        Pass       0.67      0.89      0.77      2307
   Withdrawn       0.74      0.62      0.67      1524

    accuracy                           0.66      4888
   macro avg       0.62      0.58      0.58      4888
weighted avg       0.64      0.66      0.64      4888

```

## Top 15 Feature Importances
- days_before_course_start: 4108
- mean_clicks_per_day: 2250
- clicks_week2: 2018
- std_clicks_per_day: 1810
- clicks_oucontent: 1787
- click_weekly_std: 1756
- early_vs_late_ratio: 1753
- clicks_subpage: 1688
- clicks_pre_course: 1664
- clicks_week3: 1663
- clicks_homepage: 1663
- clicks_forumng: 1567
- first_access_day: 1515
- last_access_day: 1500
- max_clicks_per_day: 1500
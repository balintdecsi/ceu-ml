# LightGBM (v2 — Improved Pipeline)

## Preprocessing
- Smart missing value handling (structural vs behavioral)
- Engineered features: engagement trends, access span, submission behavior, etc.
- One-hot encoding (fit on train only)
- Scaling: No (tree model)

## Hyperparameters
- n_estimators: 500
- max_depth: 10
- learning_rate: 0.02390376751759948
- subsample: 0.7489957156047863
- colsample_bytree: 0.6135681866731614
- min_child_samples: 20
- reg_alpha: 0.02739716567160666
- reg_lambda: 0.010085836630992925
- num_leaves: 88
- objective: multiclass
- num_class: 3
- random_state: 42
- n_jobs: -1
- verbosity: -1

## Cross-Validation Performance (5-fold Stratified)
- **Mean Accuracy**: 0.6462 ± 0.0049
- **Per-fold**: 0.6511, 0.6468, 0.6455, 0.6373, 0.6502

## Full-Train Classification Report (on last fold val set)
```text
              precision    recall  f1-score   support

        Fail       0.40      0.42      0.41      1057
        Pass       0.71      0.79      0.75      2307
   Withdrawn       0.74      0.59      0.66      1524

    accuracy                           0.65      4888
   macro avg       0.62      0.60      0.61      4888
weighted avg       0.65      0.65      0.65      4888

```

## Top 15 Feature Importances
- days_before_course_start: 6715
- mean_clicks_per_day: 4512
- clicks_week2: 3913
- std_clicks_per_day: 3790
- clicks_oucontent: 3778
- early_vs_late_ratio: 3706
- click_weekly_std: 3624
- clicks_homepage: 3467
- clicks_subpage: 3452
- clicks_forumng: 3440
- clicks_week1: 3366
- clicks_week3: 3352
- clicks_pre_course: 3293
- click_ratio_w4_w1: 3174
- click_trend_w4_w1: 3135
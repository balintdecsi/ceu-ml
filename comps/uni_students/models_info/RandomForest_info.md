# Random Forest

## Hyperparameters
- n_estimators: 100
- max_depth: 15
- min_samples_split: 5

## Validation Performance
- **Accuracy**: 0.6553

## Top 10 Feature Importances
- last_access_day: 0.0508
- clicks_week4: 0.0398
- clicks_week3: 0.0372
- mean_clicks_per_day: 0.0371
- num_active_days: 0.0361
- std_clicks_per_day: 0.0357
- num_unique_activities: 0.0352
- total_clicks: 0.0338
- clicks_homepage: 0.0316
- clicks_week2: 0.0306

## Classification Report
```text
              precision    recall  f1-score   support

        Fail       0.41      0.14      0.20      1055
        Pass       0.66      0.92      0.77      2344
   Withdrawn       0.71      0.61      0.65      1490

    accuracy                           0.66      4889
   macro avg       0.59      0.55      0.54      4889
weighted avg       0.62      0.66      0.61      4889

```

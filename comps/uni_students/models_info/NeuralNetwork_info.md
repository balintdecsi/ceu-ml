# Neural Network (MLPClassifier)

## Validation Strategy
- 3-fold stratified cross-validation
- Bootstrap confidence interval computed from the fold-level accuracies

## Best Hyperparameters
- mlp__alpha: 0.0001
- mlp__hidden_layer_sizes: (128, 64)
- mlp__learning_rate_init: 0.001

## Cross-Validation Summary
- Fold accuracies: 0.6607, 0.6544, 0.6584
- Mean accuracy: 0.6578
- Std. accuracy: 0.0032
- Bootstrap 95% CI: [0.6544, 0.6607]

## Architecture Notes
- Input features come from the engineered preprocessing pipeline.
- StandardScaler is applied inside the sklearn pipeline before the MLP.
- Early stopping is enabled to limit overfitting on the tabular feature matrix.

## Out-of-Fold Classification Report
```text
              precision    recall  f1-score   support

        Fail       0.45      0.21      0.28      5289
        Pass       0.66      0.90      0.76     11538
   Withdrawn       0.73      0.60      0.66      7617

    accuracy                           0.66     24444
   macro avg       0.61      0.57      0.57     24444
weighted avg       0.64      0.66      0.63     24444

```
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from preprocess_utils import load_and_preprocess_data
import os

def main():
    print("Loading and preprocessing data...")
    X_train_encoded, y_train, X_test_encoded, test_ids = load_and_preprocess_data('../../../data/uni_students/train.csv', '../../../data/uni_students/test.csv')
    
    # Validation split
    X_tr, X_val, y_tr, y_val = train_test_split(X_train_encoded, y_train, test_size=0.2, random_state=42)
    
    print("Training Random Forest...")
    model = RandomForestClassifier(n_estimators=100, max_depth=15, min_samples_split=5, random_state=42, n_jobs=-1)
    model.fit(X_tr, y_tr)
    
    # Validation
    val_preds = model.predict(X_val)
    val_acc = accuracy_score(y_val, val_preds)
    report = classification_report(y_val, val_preds)
    
    print(f"Validation Accuracy: {val_acc:.4f}")
    
    # Feature Importances
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]
    top_features = "\n".join([f"- {X_tr.columns[indices[i]]}: {importances[indices[i]]:.4f}" for i in range(10)])
    
    # Save info
    os.makedirs('../models_info', exist_ok=True)
    with open('../models_info/RandomForest_info.md', 'w') as f:
        f.write("# Random Forest\n\n")
        f.write("## Hyperparameters\n")
        f.write("- n_estimators: 100\n")
        f.write("- max_depth: 15\n")
        f.write("- min_samples_split: 5\n\n")
        f.write("## Validation Performance\n")
        f.write(f"- **Accuracy**: {val_acc:.4f}\n\n")
        f.write("## Top 10 Feature Importances\n")
        f.write(top_features + "\n\n")
        f.write("## Classification Report\n")
        f.write("```text\n")
        f.write(report)
        f.write("\n```\n")
        
    print("Generating predictions on test set...")
    test_preds = model.predict(X_test_encoded)
    
    submission = pd.DataFrame({'id': test_ids, 'outcome': test_preds})
    
    os.makedirs('../submissions', exist_ok=True)
    submission.to_csv('../submissions/submission_rf.csv', index=False)
    print("Submission saved to comps/uni_students/submissions/submission_rf.csv")

if __name__ == '__main__':
    main()

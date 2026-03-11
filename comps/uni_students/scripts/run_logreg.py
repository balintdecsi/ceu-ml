import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from preprocess_utils import load_and_preprocess_data
import os

def main():
    print("Loading and preprocessing data...")
    X_train_encoded, y_train, X_test_encoded, test_ids = load_and_preprocess_data('../../../data/uni_students/train.csv', '../../../data/uni_students/test.csv')
    
    # Validation split
    X_tr, X_val, y_tr, y_val = train_test_split(X_train_encoded, y_train, test_size=0.2, random_state=42)
    
    print("Training Logistic Regression...")
    model = LogisticRegression(solver='lbfgs', max_iter=1000, random_state=42)
    model.fit(X_tr, y_tr)
    
    # Validation
    val_preds = model.predict(X_val)
    val_acc = accuracy_score(y_val, val_preds)
    report = classification_report(y_val, val_preds)
    
    print(f"Validation Accuracy: {val_acc:.4f}")
    
    # Save info
    os.makedirs('../models_info', exist_ok=True)
    with open('../models_info/LogisticRegression_info.md', 'w') as f:
        f.write("# Logistic Regression\n\n")
        f.write("## Hyperparameters\n")
        f.write("- multi_class: 'multinomial'\n")
        f.write("- solver: 'lbfgs'\n")
        f.write("- max_iter: 1000\n\n")
        f.write("## Validation Performance\n")
        f.write(f"- **Accuracy**: {val_acc:.4f}\n\n")
        f.write("## Classification Report\n")
        f.write("```text\n")
        f.write(report)
        f.write("\n```\n")
        
    print("Generating predictions on test set...")
    test_preds = model.predict(X_test_encoded)
    
    submission = pd.DataFrame({'id': test_ids, 'outcome': test_preds})
    
    os.makedirs('../submissions', exist_ok=True)
    submission.to_csv('../submissions/submission_logreg.csv', index=False)
    print("Submission saved to comps/uni_students/submissions/submission_logreg.csv")

if __name__ == '__main__':
    main()

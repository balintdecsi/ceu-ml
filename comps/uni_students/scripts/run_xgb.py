import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from preprocess_utils import load_and_preprocess_data
import os

def main():
    print("Loading and preprocessing data...")
    X_train_encoded, y_train, X_test_encoded, test_ids = load_and_preprocess_data('../../../data/uni_students/train.csv', '../../../data/uni_students/test.csv')
    
    # Encode target labels for XGBoost
    le = LabelEncoder()
    y_train_encoded = le.fit_transform(y_train)
    
    # Validation split
    X_tr, X_val, y_tr, y_val = train_test_split(X_train_encoded, y_train_encoded, test_size=0.2, random_state=42)
    
    print("Training XGBoost...")
    model = XGBClassifier(
        n_estimators=100, 
        max_depth=6, 
        learning_rate=0.1, 
        objective='multi:softmax', 
        num_class=3, 
        random_state=42, 
        n_jobs=-1
    )
    model.fit(X_tr, y_tr)
    
    # Validation
    val_preds = model.predict(X_val)
    val_acc = accuracy_score(y_val, val_preds)
    
    # Decode for classification report
    y_val_decoded = le.inverse_transform(y_val)
    val_preds_decoded = le.inverse_transform(val_preds)
    report = classification_report(y_val_decoded, val_preds_decoded)
    
    print(f"Validation Accuracy: {val_acc:.4f}")
    
    # Feature Importances
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]
    top_features = "\n".join([f"- {X_tr.columns[indices[i]]}: {importances[indices[i]]:.4f}" for i in range(10)])
    
    # Save info
    os.makedirs('../models_info', exist_ok=True)
    with open('../models_info/XGBoost_info.md', 'w') as f:
        f.write("# XGBoost\n\n")
        f.write("## Hyperparameters\n")
        f.write("- n_estimators: 100\n")
        f.write("- max_depth: 6\n")
        f.write("- learning_rate: 0.1\n")
        f.write("- objective: 'multi:softmax'\n\n")
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
    test_preds_decoded = le.inverse_transform(test_preds)
    
    submission = pd.DataFrame({'id': test_ids, 'outcome': test_preds_decoded})
    
    os.makedirs('../submissions', exist_ok=True)
    submission.to_csv('../submissions/submission_xgb.csv', index=False)
    print("Submission saved to comps/uni_students/submissions/submission_xgb.csv")

if __name__ == '__main__':
    main()

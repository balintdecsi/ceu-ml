import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

def load_and_preprocess_data(train_path='data/uni_students/train.csv', test_path='data/uni_students/test.csv'):
    # Load data
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)
    
    # Separate targets and IDs
    y_train = train['outcome']
    X_train = train.drop(columns=['outcome', 'id'])
    test_ids = test['id']
    X_test = test.drop(columns=['id'])
    
    # Identify column types
    cat_cols = ['code_module', 'gender', 'region', 'highest_education', 'imd_band', 'age_band', 'disability']
    num_cols = [c for c in X_train.columns if c not in cat_cols]
    
    # Create missing indicators for scores/assessments and then fill missing values
    for df in [X_train, X_test]:
        for col in num_cols:
            if df[col].isnull().sum() > 0:
                df[f'{col}_is_missing'] = df[col].isnull().astype(int)
    
    # Re-evaluate num_cols to include missing indicators
    num_cols = [c for c in X_train.columns if c not in cat_cols]
    
    # Impute categorical (if any missing) with mode
    cat_imputer = SimpleImputer(strategy='most_frequent')
    X_train[cat_cols] = cat_imputer.fit_transform(X_train[cat_cols])
    X_test[cat_cols] = cat_imputer.transform(X_test[cat_cols])
    
    # Impute numerical with median
    num_imputer = SimpleImputer(strategy='median')
    X_train[num_cols] = num_imputer.fit_transform(X_train[num_cols])
    X_test[num_cols] = num_imputer.transform(X_test[num_cols])
    
    # Log transform highly skewed features (clicks, days late, etc)
    skewed_patterns = ['clicks', 'days', 'score']
    for col in num_cols:
        if any(p in col for p in skewed_patterns) and not col.endswith('_is_missing'):
            # Shift by min if negative to safely apply log1p
            min_val = min(X_train[col].min(), X_test[col].min())
            if min_val < 0:
                shift = abs(min_val)
                X_train[col] = np.log1p(X_train[col] + shift)
                X_test[col] = np.log1p(X_test[col] + shift)
            else:
                X_train[col] = np.log1p(X_train[col])
                X_test[col] = np.log1p(X_test[col])
                
    # One-hot encode categorical features
    # Combine to ensure same columns
    combined = pd.concat([X_train, X_test], axis=0)
    combined = pd.get_dummies(combined, columns=cat_cols, drop_first=True)
    
    X_train_encoded = combined.iloc[:len(X_train)]
    X_test_encoded = combined.iloc[len(X_train):]
    
    # Scale numerical features
    scaler = StandardScaler()
    # Find numeric features in the encoded df (all columns except the boolean dummies, actually all are numeric now)
    final_cols = [c.replace('[', '').replace(']', '').replace('<', '') for c in X_train_encoded.columns]
    X_train_encoded.columns = final_cols
    X_test_encoded.columns = final_cols
    
    X_train_encoded = pd.DataFrame(scaler.fit_transform(X_train_encoded), columns=final_cols, index=X_train.index)
    X_test_encoded = pd.DataFrame(scaler.transform(X_test_encoded), columns=final_cols, index=X_test.index)
    
    return X_train_encoded, y_train, X_test_encoded, test_ids

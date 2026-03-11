from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler


CAT_COLS = [
    'code_module',
    'gender',
    'region',
    'highest_education',
    'imd_band',
    'age_band',
    'disability',
]
# The competition notes indicate modules EEE/GGG have no assessments due in the first 28 days,
# so missing assessment features there are structural rather than behavioral.
STRUCTURAL_NO_ASSESSMENT_MODULES = {'EEE', 'GGG'}


def _validate_input_path(path):
    resolved = Path(path)
    if not resolved.exists():
        raise FileNotFoundError(
            f"Could not find '{path}'. Download the Kaggle competition train/test CSV files "
            "into data/uni_students/ or pass explicit --train-path/--test-path arguments."
        )
    return resolved


def _sanitize_columns(columns):
    return [c.replace('[', '').replace(']', '').replace('<', '').replace(' ', '_') for c in columns]


def _is_binary(series):
    unique_values = set(series.dropna().unique())
    return unique_values.issubset({0, 1})


def _add_engineered_features(df):
    df = df.copy()

    if 'code_module' in df.columns:
        structural_missing = df['code_module'].isin(STRUCTURAL_NO_ASSESSMENT_MODULES).astype(int)
        df['has_no_assessments_available'] = structural_missing

        if 'num_assessments' in df.columns:
            submitted_any_assessment = (df['num_assessments'].fillna(0) > 0).astype(int)
            df['submitted_any_assessment'] = submitted_any_assessment
            df['chose_not_to_submit'] = (
                (submitted_any_assessment == 0)
                & (~df['code_module'].isin(STRUCTURAL_NO_ASSESSMENT_MODULES))
            ).astype(int)

    score_cols = [col for col in df.columns if 'score' in col]
    if score_cols:
        all_scores_missing = df[score_cols].isna().all(axis=1).astype(int)
        df['all_scores_missing'] = all_scores_missing
        if 'has_no_assessments_available' in df.columns:
            df['behavioral_score_missing'] = (
                (all_scores_missing == 1) & (df['has_no_assessments_available'] == 0)
            ).astype(int)

    return df


def _build_missing_indicators(X_train, X_test, num_cols):
    cols_with_missing = [col for col in num_cols if X_train[col].isna().any() or X_test[col].isna().any()]
    for df in [X_train, X_test]:
        for col in cols_with_missing:
            df[f'{col}_is_missing'] = df[col].isna().astype(int)


def _log_transform_numeric(X_train, X_test, num_cols):
    skewed_patterns = ['clicks', 'days', 'score']
    for col in num_cols:
        if col.endswith('_is_missing') or _is_binary(X_train[col]):
            continue
        if not any(pattern in col for pattern in skewed_patterns):
            continue

        min_val = min(X_train[col].min(), X_test[col].min())
        shift = abs(min_val) if min_val < 0 else 0
        X_train[col] = np.log1p(X_train[col] + shift)
        X_test[col] = np.log1p(X_test[col] + shift)


def _encode_and_align(X_train, X_test, cat_cols):
    X_train_encoded = pd.get_dummies(X_train, columns=cat_cols, drop_first=True)
    X_test_encoded = pd.get_dummies(X_test, columns=cat_cols, drop_first=True)
    X_test_encoded = X_test_encoded.reindex(columns=X_train_encoded.columns, fill_value=0)

    clean_columns = _sanitize_columns(X_train_encoded.columns)
    X_train_encoded.columns = clean_columns
    X_test_encoded.columns = clean_columns

    return X_train_encoded, X_test_encoded


def _scale_continuous_columns(X_train, X_test):
    continuous_cols = [col for col in X_train.columns if not _is_binary(X_train[col])]
    if not continuous_cols:
        return X_train, X_test

    scaler = StandardScaler()
    X_train = X_train.copy()
    X_test = X_test.copy()
    X_train[continuous_cols] = scaler.fit_transform(X_train[continuous_cols])
    X_test[continuous_cols] = scaler.transform(X_test[continuous_cols])
    return X_train, X_test


def load_and_preprocess_data(
    train_path='data/uni_students/train.csv',
    test_path='data/uni_students/test.csv',
    *,
    scale_numeric=True,
    add_engineered_features=True,
):
    train = pd.read_csv(_validate_input_path(train_path))
    test = pd.read_csv(_validate_input_path(test_path))

    y_train = train['outcome']
    X_train = train.drop(columns=['outcome', 'id'])
    test_ids = test['id']
    X_test = test.drop(columns=['id'])

    if add_engineered_features:
        X_train = _add_engineered_features(X_train)
        X_test = _add_engineered_features(X_test)

    cat_cols = [col for col in CAT_COLS if col in X_train.columns]
    num_cols = [col for col in X_train.columns if col not in cat_cols]

    _build_missing_indicators(X_train, X_test, num_cols)

    num_cols = [col for col in X_train.columns if col not in cat_cols]

    if cat_cols:
        cat_imputer = SimpleImputer(strategy='most_frequent')
        X_train[cat_cols] = cat_imputer.fit_transform(X_train[cat_cols])
        X_test[cat_cols] = cat_imputer.transform(X_test[cat_cols])

    num_imputer = SimpleImputer(strategy='median')
    X_train[num_cols] = num_imputer.fit_transform(X_train[num_cols])
    X_test[num_cols] = num_imputer.transform(X_test[num_cols])

    _log_transform_numeric(X_train, X_test, num_cols)

    X_train_encoded, X_test_encoded = _encode_and_align(X_train, X_test, cat_cols)

    if scale_numeric:
        X_train_encoded, X_test_encoded = _scale_continuous_columns(X_train_encoded, X_test_encoded)

    return X_train_encoded, y_train, X_test_encoded, test_ids

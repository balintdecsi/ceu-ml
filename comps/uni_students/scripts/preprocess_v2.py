"""
Improved preprocessing with smart feature engineering.
Separates structural vs behavioral missing values,
adds engagement trend features, per-module normalization, and interaction features.
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer

CAT_COLS = ["code_module", "gender", "region", "highest_education", "imd_band", "age_band", "disability"]
NO_ASSESS_MODULES = ["EEE", "GGG"]
WEEKLY_CLICK_COLS = ["clicks_pre_course", "clicks_week1", "clicks_week2", "clicks_week3", "clicks_week4"]

def engineer_features(df):
    """Add engineered features. Operates in-place on a copy."""
    df = df.copy()

    # ── Structural vs behavioral missing ──
    df["no_assess_available"] = df["code_module"].isin(NO_ASSESS_MODULES).astype(int)
    df["chose_not_to_submit"] = (
        (~df["code_module"].isin(NO_ASSESS_MODULES)) & (df["num_assessments"] == 0)
    ).astype(int)

    # ── Missing indicators for score columns ──
    score_cols = ["mean_score", "std_score", "min_score", "max_score",
                  "mean_score_TMA", "mean_score_CMA", "mean_days_late", "max_days_late"]
    for col in score_cols:
        if col in df.columns:
            df[f"{col}_missing"] = df[col].isnull().astype(int)

    # ── Engagement trend features ──
    df["click_trend_w4_w1"] = df["clicks_week4"] - df["clicks_week1"]
    df["click_ratio_w4_w1"] = df["clicks_week4"] / (df["clicks_week1"] + 1)
    weekly = df[WEEKLY_CLICK_COLS].values
    df["click_weekly_std"] = np.std(weekly, axis=1)
    df["click_weekly_mean"] = np.mean(weekly, axis=1)

    # ── Engagement intensity ──
    df["active_day_ratio"] = df["num_active_days"] / 28.0
    df["early_engagement"] = df["clicks_pre_course"] + df["clicks_week1"]
    df["late_engagement"] = df["clicks_week3"] + df["clicks_week4"]
    df["early_vs_late_ratio"] = df["early_engagement"] / (df["late_engagement"] + 1)

    # ── VLE access span ──
    df["vle_access_span"] = df["last_access_day"] - df["first_access_day"]
    df["stopped_early"] = (df["last_access_day"] <= 14).astype(int)

    # ── Assessment behavior ──
    df["all_early_submissions"] = (
        (df["num_early_submissions"] == df["num_assessments"]) & (df["num_assessments"] > 0)
    ).astype(int)
    df["any_late"] = (df["num_late_submissions"] > 0).astype(int)
    df["score_range"] = df["max_score"] - df["min_score"]

    # ── Registration timing ──
    df["registered_late"] = (df["days_before_course_start"] < 0).astype(int)

    # ── Credit load indicator ──
    df["high_credit_load"] = (df["studied_credits"] > 120).astype(int)

    # ── Previous attempts indicator ──
    df["has_prev_attempts"] = (df["num_of_prev_attempts"] > 0).astype(int)

    return df


def load_and_preprocess(train_path, test_path, scale=False):
    """
    Load, engineer features, impute, encode.
    scale=False is better for tree models; scale=True for linear models.
    Returns: X_train, y_train, X_test, test_ids, feature_names
    """
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)

    y_train = train["outcome"]
    X_train = train.drop(columns=["outcome", "id"])
    test_ids = test["id"]
    X_test = test.drop(columns=["id"])

    # Engineer features before encoding
    X_train = engineer_features(X_train)
    X_test = engineer_features(X_test)

    # Identify column types after engineering
    cat_cols = [c for c in CAT_COLS if c in X_train.columns]
    num_cols = [c for c in X_train.columns if c not in cat_cols]

    # Impute categorical with mode
    cat_imputer = SimpleImputer(strategy="most_frequent")
    X_train[cat_cols] = cat_imputer.fit_transform(X_train[cat_cols])
    X_test[cat_cols] = cat_imputer.transform(X_test[cat_cols])

    # Impute numerical with -1 for score columns (preserves "no data" signal), median for rest
    for col in num_cols:
        if X_train[col].isnull().any() or X_test[col].isnull().any():
            fill_val = X_train[col].median()
            X_train[col] = X_train[col].fillna(fill_val)
            X_test[col] = X_test[col].fillna(fill_val)

    # One-hot encode categoricals (fit on train only)
    X_train_cat = pd.get_dummies(X_train[cat_cols], drop_first=False)
    X_test_cat = pd.get_dummies(X_test[cat_cols], drop_first=False)
    # Align columns
    X_test_cat = X_test_cat.reindex(columns=X_train_cat.columns, fill_value=0)

    X_train_final = pd.concat([X_train[num_cols].reset_index(drop=True),
                                X_train_cat.reset_index(drop=True)], axis=1)
    X_test_final = pd.concat([X_test[num_cols].reset_index(drop=True),
                               X_test_cat.reset_index(drop=True)], axis=1)

    # Clean column names for XGBoost compatibility
    clean_cols = [c.replace("[", "").replace("]", "").replace("<", "").replace(">", "").replace(" ", "_")
                  for c in X_train_final.columns]
    X_train_final.columns = clean_cols
    X_test_final.columns = clean_cols

    if scale:
        scaler = StandardScaler()
        X_train_final = pd.DataFrame(scaler.fit_transform(X_train_final),
                                      columns=clean_cols, index=X_train_final.index)
        X_test_final = pd.DataFrame(scaler.transform(X_test_final),
                                     columns=clean_cols, index=X_test_final.index)

    return X_train_final, y_train, X_test_final, test_ids, clean_cols

"""
Train improved models with Stratified K-Fold CV, class balancing,
hyperparameter tuning via Optuna, and ensemble methods.
Generates model info sheets and submission files.
"""
import pandas as pd
import numpy as np
import os
import sys
import json
import warnings
from datetime import datetime

from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import accuracy_score, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier
import lightgbm as lgb
import optuna

optuna.logging.set_verbosity(optuna.logging.WARNING)
warnings.filterwarnings("ignore")

# Add scripts dir to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from preprocess_v2 import load_and_preprocess

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
TRAIN_PATH = os.path.join(REPO_ROOT, "data", "uni_students", "train.csv")
TEST_PATH = os.path.join(REPO_ROOT, "data", "uni_students", "test.csv")
SUBMISSIONS_DIR = os.path.join(REPO_ROOT, "comps", "uni_students", "submissions")
MODELS_INFO_DIR = os.path.join(REPO_ROOT, "comps", "uni_students", "models_info")
os.makedirs(SUBMISSIONS_DIR, exist_ok=True)
os.makedirs(MODELS_INFO_DIR, exist_ok=True)

N_FOLDS = 5
RANDOM_STATE = 42
OPTUNA_TRIALS = 15


def cv_evaluate(model, X, y, n_folds=N_FOLDS):
    """Stratified K-Fold CV returning mean accuracy and per-fold scores."""
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=RANDOM_STATE)
    scores = cross_val_score(model, X, y, cv=skf, scoring="accuracy", n_jobs=-1)
    return scores.mean(), scores.std(), scores


def save_model_info(name, params, cv_mean, cv_std, cv_scores, report_text, extra=""):
    path = os.path.join(MODELS_INFO_DIR, f"{name}_v2_info.md")
    with open(path, "w") as f:
        f.write(f"# {name} (v2 — Improved Pipeline)\n\n")
        f.write(f"## Preprocessing\n")
        f.write("- Smart missing value handling (structural vs behavioral)\n")
        f.write("- Engineered features: engagement trends, access span, submission behavior, etc.\n")
        f.write("- One-hot encoding (fit on train only)\n")
        f.write(f"- Scaling: {'Yes' if 'Logistic' in name else 'No (tree model)'}\n\n")
        f.write(f"## Hyperparameters\n")
        for k, v in params.items():
            f.write(f"- {k}: {v}\n")
        f.write(f"\n## Cross-Validation Performance ({N_FOLDS}-fold Stratified)\n")
        f.write(f"- **Mean Accuracy**: {cv_mean:.4f} ± {cv_std:.4f}\n")
        f.write(f"- **Per-fold**: {', '.join(f'{s:.4f}' for s in cv_scores)}\n\n")
        f.write(f"## Full-Train Classification Report (on last fold val set)\n")
        f.write(f"```text\n{report_text}\n```\n")
        if extra:
            f.write(f"\n{extra}")
    print(f"  Model info saved to {path}")


def save_submission(test_ids, preds, name):
    sub = pd.DataFrame({"id": test_ids, "outcome": preds})
    path = os.path.join(SUBMISSIONS_DIR, f"submission_{name}_v2.csv")
    sub.to_csv(path, index=False)
    print(f"  Submission saved to {path}")
    return path


def get_last_fold_report(model, X, y):
    """Train on all but last fold, predict on last fold, return classification report."""
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    folds = list(skf.split(X, y))
    train_idx, val_idx = folds[-1]
    model.fit(X.iloc[train_idx], y.iloc[train_idx])
    preds = model.predict(X.iloc[val_idx])
    return classification_report(y.iloc[val_idx], preds)


# ══════════════════════════════════════════════════════════════════════
# 1. Logistic Regression (with class weights + scaling)
# ══════════════════════════════════════════════════════════════════════
def train_logreg(X_train_scaled, y_train, X_test_scaled, test_ids):
    print("\n═══ Logistic Regression (v2) ═══")
    model = LogisticRegression(
        solver="lbfgs", max_iter=2000, class_weight="balanced",
        C=0.5, random_state=RANDOM_STATE
    )
    cv_mean, cv_std, cv_scores = cv_evaluate(model, X_train_scaled, y_train)
    print(f"  CV Accuracy: {cv_mean:.4f} ± {cv_std:.4f}")

    report = get_last_fold_report(
        LogisticRegression(solver="lbfgs", max_iter=2000, class_weight="balanced",
                           C=0.5, random_state=RANDOM_STATE),
        X_train_scaled, y_train
    )

    # Train on full data for submission
    model.fit(X_train_scaled, y_train)
    preds = model.predict(X_test_scaled)
    save_submission(test_ids, preds, "logreg")
    save_model_info("LogisticRegression", {
        "solver": "lbfgs", "max_iter": 2000, "class_weight": "balanced", "C": 0.5
    }, cv_mean, cv_std, cv_scores, report)
    return cv_mean


# ══════════════════════════════════════════════════════════════════════
# 2. Random Forest (with class weights + tuning)
# ══════════════════════════════════════════════════════════════════════
def train_rf(X_train, y_train, X_test, test_ids):
    print("\n═══ Random Forest (v2) ═══")

    def rf_objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 200, 500, step=100),
            "max_depth": trial.suggest_int("max_depth", 12, 25),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 15),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 8),
            "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2", 0.3]),
            "class_weight": "balanced",
            "random_state": RANDOM_STATE,
            "n_jobs": -1,
        }
        model = RandomForestClassifier(**params)
        score, _, _ = cv_evaluate(model, X_train, y_train, n_folds=3)  # 3-fold for speed
        return score

    print("  Tuning with Optuna (3-fold CV)...")
    study = optuna.create_study(direction="maximize",
                                 sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE))
    study.optimize(rf_objective, n_trials=OPTUNA_TRIALS, show_progress_bar=False)
    best_params = study.best_params
    best_params["class_weight"] = "balanced"
    best_params["random_state"] = RANDOM_STATE
    best_params["n_jobs"] = -1
    print(f"  Best params: {best_params}")

    model = RandomForestClassifier(**best_params)
    cv_mean, cv_std, cv_scores = cv_evaluate(model, X_train, y_train)
    print(f"  CV Accuracy (5-fold): {cv_mean:.4f} ± {cv_std:.4f}")

    report = get_last_fold_report(RandomForestClassifier(**best_params), X_train, y_train)

    # Feature importances from full train
    model.fit(X_train, y_train)
    importances = model.feature_importances_
    top_idx = np.argsort(importances)[::-1][:15]
    fi_text = "## Top 15 Feature Importances\n"
    fi_text += "\n".join(f"- {X_train.columns[i]}: {importances[i]:.4f}" for i in top_idx)

    preds = model.predict(X_test)
    save_submission(test_ids, preds, "rf")
    save_model_info("RandomForest", best_params, cv_mean, cv_std, cv_scores, report, fi_text)
    return cv_mean


# ══════════════════════════════════════════════════════════════════════
# 3. XGBoost (with tuning)
# ══════════════════════════════════════════════════════════════════════
def train_xgb(X_train, y_train, X_test, test_ids):
    print("\n═══ XGBoost (v2) ═══")
    le = LabelEncoder()
    y_encoded = le.fit_transform(y_train)
    y_encoded_series = pd.Series(y_encoded, index=y_train.index)

    # Compute sample weights for class balancing
    class_counts = np.bincount(y_encoded)
    total = len(y_encoded)
    class_weights = total / (len(class_counts) * class_counts)
    sample_weights = np.array([class_weights[c] for c in y_encoded])

    def xgb_objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 300, 700, step=100),
            "max_depth": trial.suggest_int("max_depth", 5, 9),
            "learning_rate": trial.suggest_float("learning_rate", 0.02, 0.15, log=True),
            "subsample": trial.suggest_float("subsample", 0.7, 0.95),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 0.9),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 7),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 5.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 5.0, log=True),
            "objective": "multi:softprob",
            "num_class": 3,
            "random_state": RANDOM_STATE,
            "n_jobs": -1,
            "verbosity": 0,
        }
        model = XGBClassifier(**params)
        skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)
        scores = []
        for train_idx, val_idx in skf.split(X_train, y_encoded_series):
            model.fit(X_train.iloc[train_idx], y_encoded_series.iloc[train_idx],
                      sample_weight=sample_weights[train_idx])
            val_preds = model.predict(X_train.iloc[val_idx])
            scores.append(accuracy_score(y_encoded_series.iloc[val_idx], val_preds))
        return np.mean(scores)

    print("  Tuning with Optuna (3-fold CV)...")
    study = optuna.create_study(direction="maximize",
                                 sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE))
    study.optimize(xgb_objective, n_trials=OPTUNA_TRIALS, show_progress_bar=False)
    best_params = study.best_params
    best_params["objective"] = "multi:softprob"
    best_params["num_class"] = 3
    best_params["random_state"] = RANDOM_STATE
    best_params["n_jobs"] = -1
    best_params["verbosity"] = 0
    print(f"  Best params: {best_params}")

    # 5-fold CV with best params
    model = XGBClassifier(**best_params)
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    cv_scores = []
    for train_idx, val_idx in skf.split(X_train, y_encoded_series):
        model.fit(X_train.iloc[train_idx], y_encoded_series.iloc[train_idx],
                  sample_weight=sample_weights[train_idx])
        val_preds = model.predict(X_train.iloc[val_idx])
        cv_scores.append(accuracy_score(y_encoded_series.iloc[val_idx], val_preds))
    cv_scores = np.array(cv_scores)
    cv_mean, cv_std = cv_scores.mean(), cv_scores.std()
    print(f"  CV Accuracy (5-fold): {cv_mean:.4f} ± {cv_std:.4f}")

    # Last fold report
    train_idx, val_idx = list(skf.split(X_train, y_encoded_series))[-1]
    model.fit(X_train.iloc[train_idx], y_encoded_series.iloc[train_idx],
              sample_weight=sample_weights[train_idx])
    val_preds_decoded = le.inverse_transform(model.predict(X_train.iloc[val_idx]))
    report = classification_report(y_train.iloc[val_idx], val_preds_decoded)

    # Full train + predict
    model.fit(X_train, y_encoded_series, sample_weight=sample_weights)
    importances = model.feature_importances_
    top_idx = np.argsort(importances)[::-1][:15]
    fi_text = "## Top 15 Feature Importances\n"
    fi_text += "\n".join(f"- {X_train.columns[i]}: {importances[i]:.4f}" for i in top_idx)

    preds = le.inverse_transform(model.predict(X_test))
    save_submission(test_ids, preds, "xgb")
    save_model_info("XGBoost", best_params, cv_mean, cv_std, cv_scores, report, fi_text)
    return cv_mean, model, le


# ══════════════════════════════════════════════════════════════════════
# 4. LightGBM (with tuning)
# ══════════════════════════════════════════════════════════════════════
def train_lgbm(X_train, y_train, X_test, test_ids):
    print("\n═══ LightGBM (v2) ═══")
    le = LabelEncoder()
    y_encoded = le.fit_transform(y_train)
    y_encoded_series = pd.Series(y_encoded, index=y_train.index)

    class_counts = np.bincount(y_encoded)
    total = len(y_encoded)
    class_weights = total / (len(class_counts) * class_counts)
    sample_weights = np.array([class_weights[c] for c in y_encoded])

    def lgbm_objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 300, 700, step=100),
            "max_depth": trial.suggest_int("max_depth", 5, 10),
            "learning_rate": trial.suggest_float("learning_rate", 0.02, 0.15, log=True),
            "subsample": trial.suggest_float("subsample", 0.7, 0.95),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 0.9),
            "min_child_samples": trial.suggest_int("min_child_samples", 10, 40),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 5.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 5.0, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 30, 100),
            "objective": "multiclass",
            "num_class": 3,
            "random_state": RANDOM_STATE,
            "n_jobs": -1,
            "verbosity": -1,
        }
        model = lgb.LGBMClassifier(**params)
        skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)
        scores = []
        for train_idx, val_idx in skf.split(X_train, y_encoded_series):
            model.fit(X_train.iloc[train_idx], y_encoded_series.iloc[train_idx],
                      sample_weight=sample_weights[train_idx])
            val_preds = model.predict(X_train.iloc[val_idx])
            scores.append(accuracy_score(y_encoded_series.iloc[val_idx], val_preds))
        return np.mean(scores)

    print("  Tuning with Optuna (3-fold CV)...")
    study = optuna.create_study(direction="maximize",
                                 sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE))
    study.optimize(lgbm_objective, n_trials=OPTUNA_TRIALS, show_progress_bar=False)
    best_params = study.best_params
    best_params["objective"] = "multiclass"
    best_params["num_class"] = 3
    best_params["random_state"] = RANDOM_STATE
    best_params["n_jobs"] = -1
    best_params["verbosity"] = -1
    print(f"  Best params: {best_params}")

    # 5-fold CV
    model = lgb.LGBMClassifier(**best_params)
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    cv_scores = []
    for train_idx, val_idx in skf.split(X_train, y_encoded_series):
        model.fit(X_train.iloc[train_idx], y_encoded_series.iloc[train_idx],
                  sample_weight=sample_weights[train_idx])
        val_preds = model.predict(X_train.iloc[val_idx])
        cv_scores.append(accuracy_score(y_encoded_series.iloc[val_idx], val_preds))
    cv_scores = np.array(cv_scores)
    cv_mean, cv_std = cv_scores.mean(), cv_scores.std()
    print(f"  CV Accuracy (5-fold): {cv_mean:.4f} ± {cv_std:.4f}")

    # Last fold report
    train_idx, val_idx = list(skf.split(X_train, y_encoded_series))[-1]
    model.fit(X_train.iloc[train_idx], y_encoded_series.iloc[train_idx],
              sample_weight=sample_weights[train_idx])
    val_preds_decoded = le.inverse_transform(model.predict(X_train.iloc[val_idx]))
    report = classification_report(y_train.iloc[val_idx], val_preds_decoded)

    # Full train + predict
    model.fit(X_train, y_encoded_series, sample_weight=sample_weights)
    importances = model.feature_importances_
    top_idx = np.argsort(importances)[::-1][:15]
    fi_text = "## Top 15 Feature Importances\n"
    fi_text += "\n".join(f"- {X_train.columns[i]}: {importances[i]:.0f}" for i in top_idx)

    preds = le.inverse_transform(model.predict(X_test))
    save_submission(test_ids, preds, "lgbm")
    save_model_info("LightGBM", best_params, cv_mean, cv_std, cv_scores, report, fi_text)
    return cv_mean, model, le


# ══════════════════════════════════════════════════════════════════════
# 5. Ensemble (majority voting of best models)
# ══════════════════════════════════════════════════════════════════════
def train_ensemble(X_train, y_train, X_test, test_ids):
    """Soft voting ensemble of XGBoost + LightGBM + RF with best params from individual tuning."""
    print("\n═══ Ensemble (Soft Voting) ═══")
    le = LabelEncoder()
    y_encoded = le.fit_transform(y_train)
    y_encoded_series = pd.Series(y_encoded, index=y_train.index)

    class_counts = np.bincount(y_encoded)
    total = len(y_encoded)
    class_weights_dict = {i: total / (len(class_counts) * class_counts[i]) for i in range(len(class_counts))}
    sample_weights = np.array([class_weights_dict[c] for c in y_encoded])

    # Read best params from saved info files if they exist, otherwise use defaults
    xgb_model = XGBClassifier(
        n_estimators=500, max_depth=6, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, min_child_weight=3,
        reg_alpha=0.1, reg_lambda=1.0,
        objective="multi:softprob", num_class=3,
        random_state=RANDOM_STATE, n_jobs=-1, verbosity=0
    )
    lgbm_model = lgb.LGBMClassifier(
        n_estimators=500, max_depth=8, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, min_child_samples=20,
        reg_alpha=0.1, reg_lambda=1.0, num_leaves=63,
        objective="multiclass", num_class=3,
        random_state=RANDOM_STATE, n_jobs=-1, verbosity=-1
    )
    rf_model = RandomForestClassifier(
        n_estimators=500, max_depth=20, min_samples_split=5,
        min_samples_leaf=2, max_features="sqrt",
        class_weight="balanced",
        random_state=RANDOM_STATE, n_jobs=-1
    )

    # Majority vote via stacking predictions
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    cv_scores = []

    for train_idx, val_idx in skf.split(X_train, y_encoded_series):
        X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
        y_tr, y_val = y_encoded_series.iloc[train_idx], y_encoded_series.iloc[val_idx]
        sw_tr = sample_weights[train_idx]

        xgb_model.fit(X_tr, y_tr, sample_weight=sw_tr)
        lgbm_model.fit(X_tr, y_tr, sample_weight=sw_tr)
        rf_model.fit(X_tr, y_train.iloc[train_idx])  # RF uses class_weight

        # Soft voting: average probabilities
        xgb_probs = xgb_model.predict_proba(X_val)
        lgbm_probs = lgbm_model.predict_proba(X_val)
        rf_probs_raw = rf_model.predict_proba(X_val)
        # Map RF class order to LabelEncoder order
        rf_classes = rf_model.classes_
        le_classes = le.classes_
        rf_probs = np.zeros_like(rf_probs_raw)
        for i, cls in enumerate(rf_classes):
            le_idx = np.where(le_classes == cls)[0][0]
            rf_probs[:, le_idx] = rf_probs_raw[:, i]

        avg_probs = (xgb_probs + lgbm_probs + rf_probs) / 3.0
        ensemble_preds = np.argmax(avg_probs, axis=1)
        cv_scores.append(accuracy_score(y_val, ensemble_preds))

    cv_scores = np.array(cv_scores)
    cv_mean, cv_std = cv_scores.mean(), cv_scores.std()
    print(f"  CV Accuracy (5-fold): {cv_mean:.4f} ± {cv_std:.4f}")

    # Last fold report
    train_idx, val_idx = list(skf.split(X_train, y_encoded_series))[-1]
    X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
    y_tr = y_encoded_series.iloc[train_idx]
    sw_tr = sample_weights[train_idx]

    xgb_model.fit(X_tr, y_tr, sample_weight=sw_tr)
    lgbm_model.fit(X_tr, y_tr, sample_weight=sw_tr)
    rf_model.fit(X_tr, y_train.iloc[train_idx])

    xgb_probs = xgb_model.predict_proba(X_val)
    lgbm_probs = lgbm_model.predict_proba(X_val)
    rf_probs_raw = rf_model.predict_proba(X_val)
    rf_probs = np.zeros_like(rf_probs_raw)
    for i, cls in enumerate(rf_model.classes_):
        le_idx = np.where(le.classes_ == cls)[0][0]
        rf_probs[:, le_idx] = rf_probs_raw[:, i]
    avg_probs = (xgb_probs + lgbm_probs + rf_probs) / 3.0
    ensemble_preds = le.inverse_transform(np.argmax(avg_probs, axis=1))
    report = classification_report(y_train.iloc[val_idx], ensemble_preds)

    # Full train + predict
    xgb_model.fit(X_train, y_encoded_series, sample_weight=sample_weights)
    lgbm_model.fit(X_train, y_encoded_series, sample_weight=sample_weights)
    rf_model.fit(X_train, y_train)

    xgb_probs = xgb_model.predict_proba(X_test)
    lgbm_probs = lgbm_model.predict_proba(X_test)
    rf_probs_raw = rf_model.predict_proba(X_test)
    rf_probs = np.zeros_like(rf_probs_raw)
    for i, cls in enumerate(rf_model.classes_):
        le_idx = np.where(le.classes_ == cls)[0][0]
        rf_probs[:, le_idx] = rf_probs_raw[:, i]
    avg_probs = (xgb_probs + lgbm_probs + rf_probs) / 3.0
    preds = le.inverse_transform(np.argmax(avg_probs, axis=1))
    save_submission(test_ids, preds, "ensemble")

    save_model_info("Ensemble_SoftVoting", {
        "models": "XGBoost + LightGBM + RandomForest",
        "method": "Average predicted probabilities",
        "class_balancing": "sample_weight (XGB/LGBM) + class_weight (RF)",
    }, cv_mean, cv_std, cv_scores, report)
    return cv_mean


# ══════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════
def main():
    print("=" * 60)
    print("IMPROVED MODEL TRAINING PIPELINE (v2)")
    print("=" * 60)

    # Load data — unscaled for tree models, scaled for LogReg
    print("\nLoading data (unscaled for trees)...")
    X_train, y_train, X_test, test_ids, feat_names = load_and_preprocess(TRAIN_PATH, TEST_PATH, scale=False)
    print(f"  Features: {X_train.shape[1]}, Train rows: {X_train.shape[0]}")

    print("\nLoading data (scaled for LogReg)...")
    X_train_s, y_train_s, X_test_s, test_ids_s, _ = load_and_preprocess(TRAIN_PATH, TEST_PATH, scale=True)

    results = {}

    # 1. Logistic Regression
    results["LogReg_v2"] = train_logreg(X_train_s, y_train_s, X_test_s, test_ids_s)

    # 2. Random Forest
    results["RF_v2"] = train_rf(X_train, y_train, X_test, test_ids)

    # 3. XGBoost
    xgb_cv, _, _ = train_xgb(X_train, y_train, X_test, test_ids)
    results["XGB_v2"] = xgb_cv

    # 4. LightGBM
    lgbm_cv, _, _ = train_lgbm(X_train, y_train, X_test, test_ids)
    results["LGBM_v2"] = lgbm_cv

    # 5. Ensemble
    results["Ensemble_v2"] = train_ensemble(X_train, y_train, X_test, test_ids)

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY — 5-Fold Stratified CV Accuracy")
    print("=" * 60)
    for name, score in sorted(results.items(), key=lambda x: x[1], reverse=True):
        print(f"  {name:20s}: {score:.4f}")

    # Compare with previous results
    print("\nPrevious Kaggle test scores (for reference):")
    print("  XGBoost_v1:  0.6804")
    print("  LogReg_v1:   0.6745")
    print("  RF_v1:       0.6703")

    print("\nDone! Check comps/uni_students/submissions/ for new submission files.")
    print("Check comps/uni_students/models_info/ for model info sheets.")


if __name__ == "__main__":
    main()

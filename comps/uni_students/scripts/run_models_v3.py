"""
v3: Accuracy-optimized models — improved features WITHOUT class balancing.
Class balancing hurts accuracy on imbalanced data since the metric rewards
majority class prediction. This version maximizes accuracy directly.
"""
import pandas as pd
import numpy as np
import os
import sys
import warnings

from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import accuracy_score, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier
import lightgbm as lgb
import optuna

optuna.logging.set_verbosity(optuna.logging.WARNING)
warnings.filterwarnings("ignore")

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
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=RANDOM_STATE)
    scores = cross_val_score(model, X, y, cv=skf, scoring="accuracy", n_jobs=-1)
    return scores.mean(), scores.std(), scores


def save_model_info(name, params, cv_mean, cv_std, cv_scores, report_text, extra=""):
    path = os.path.join(MODELS_INFO_DIR, f"{name}_v3_info.md")
    with open(path, "w") as f:
        f.write(f"# {name} (v3 — Accuracy-Optimized)\n\n")
        f.write("## Preprocessing\n")
        f.write("- Smart missing value handling (structural vs behavioral)\n")
        f.write("- Engineered features: engagement trends, access span, submission behavior\n")
        f.write("- One-hot encoding (fit on train only)\n")
        f.write(f"- Scaling: {'Yes' if 'Logistic' in name else 'No (tree model)'}\n")
        f.write("- **No class balancing** (accuracy metric rewards majority class)\n\n")
        f.write("## Hyperparameters\n")
        for k, v in params.items():
            f.write(f"- {k}: {v}\n")
        f.write(f"\n## Cross-Validation Performance ({N_FOLDS}-fold Stratified)\n")
        f.write(f"- **Mean Accuracy**: {cv_mean:.4f} ± {cv_std:.4f}\n")
        f.write(f"- **Per-fold**: {', '.join(f'{s:.4f}' for s in cv_scores)}\n\n")
        f.write(f"## Classification Report (last fold val set)\n")
        f.write(f"```text\n{report_text}\n```\n")
        if extra:
            f.write(f"\n{extra}")
    print(f"  Info saved: {os.path.basename(path)}")


def save_submission(test_ids, preds, name):
    sub = pd.DataFrame({"id": test_ids, "outcome": preds})
    path = os.path.join(SUBMISSIONS_DIR, f"submission_{name}_v3.csv")
    sub.to_csv(path, index=False)
    print(f"  Submission: {os.path.basename(path)}")


def get_last_fold_report(model, X, y):
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    folds = list(skf.split(X, y))
    train_idx, val_idx = folds[-1]
    model.fit(X.iloc[train_idx], y.iloc[train_idx])
    return classification_report(y.iloc[val_idx], model.predict(X.iloc[val_idx]))


def train_logreg(X_train, y_train, X_test, test_ids):
    print("\n═══ Logistic Regression (v3 — no balancing) ═══")
    # Tune C
    def objective(trial):
        C = trial.suggest_float("C", 0.01, 10.0, log=True)
        model = LogisticRegression(solver="lbfgs", max_iter=2000, C=C, random_state=RANDOM_STATE)
        score, _, _ = cv_evaluate(model, X_train, y_train, n_folds=3)
        return score

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE))
    study.optimize(objective, n_trials=OPTUNA_TRIALS)
    best_C = study.best_params["C"]

    model = LogisticRegression(solver="lbfgs", max_iter=2000, C=best_C, random_state=RANDOM_STATE)
    cv_mean, cv_std, cv_scores = cv_evaluate(model, X_train, y_train)
    print(f"  CV: {cv_mean:.4f} ± {cv_std:.4f} (C={best_C:.4f})")

    report = get_last_fold_report(
        LogisticRegression(solver="lbfgs", max_iter=2000, C=best_C, random_state=RANDOM_STATE),
        X_train, y_train)

    model.fit(X_train, y_train)
    save_submission(test_ids, model.predict(X_test), "logreg")
    save_model_info("LogisticRegression", {"solver": "lbfgs", "max_iter": 2000, "C": f"{best_C:.4f}"},
                    cv_mean, cv_std, cv_scores, report)
    return cv_mean


def train_xgb(X_train, y_train, X_test, test_ids):
    print("\n═══ XGBoost (v3 — no balancing) ═══")
    le = LabelEncoder()
    y_enc = pd.Series(le.fit_transform(y_train), index=y_train.index)

    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 300, 700, step=100),
            "max_depth": trial.suggest_int("max_depth", 5, 9),
            "learning_rate": trial.suggest_float("learning_rate", 0.02, 0.15, log=True),
            "subsample": trial.suggest_float("subsample", 0.7, 0.95),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 0.9),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 7),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 5.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 5.0, log=True),
            "objective": "multi:softprob", "num_class": 3,
            "random_state": RANDOM_STATE, "n_jobs": -1, "verbosity": 0,
        }
        model = XGBClassifier(**params)
        skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)
        scores = []
        for ti, vi in skf.split(X_train, y_enc):
            model.fit(X_train.iloc[ti], y_enc.iloc[ti])
            scores.append(accuracy_score(y_enc.iloc[vi], model.predict(X_train.iloc[vi])))
        return np.mean(scores)

    print("  Tuning...")
    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE))
    study.optimize(objective, n_trials=OPTUNA_TRIALS)
    bp = study.best_params
    bp.update({"objective": "multi:softprob", "num_class": 3, "random_state": RANDOM_STATE,
               "n_jobs": -1, "verbosity": 0})
    print(f"  Best: n_est={bp['n_estimators']}, depth={bp['max_depth']}, lr={bp['learning_rate']:.4f}")

    model = XGBClassifier(**bp)
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    cv_scores = []
    for ti, vi in skf.split(X_train, y_enc):
        model.fit(X_train.iloc[ti], y_enc.iloc[ti])
        cv_scores.append(accuracy_score(y_enc.iloc[vi], model.predict(X_train.iloc[vi])))
    cv_scores = np.array(cv_scores)
    print(f"  CV: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

    # Last fold report
    ti, vi = list(skf.split(X_train, y_enc))[-1]
    model.fit(X_train.iloc[ti], y_enc.iloc[ti])
    report = classification_report(y_train.iloc[vi], le.inverse_transform(model.predict(X_train.iloc[vi])))

    model.fit(X_train, y_enc)
    importances = model.feature_importances_
    top_idx = np.argsort(importances)[::-1][:15]
    fi = "## Top 15 Feature Importances\n" + "\n".join(
        f"- {X_train.columns[i]}: {importances[i]:.4f}" for i in top_idx)

    save_submission(test_ids, le.inverse_transform(model.predict(X_test)), "xgb")
    save_model_info("XGBoost", bp, cv_scores.mean(), cv_scores.std(), cv_scores, report, fi)
    return cv_scores.mean(), model, le


def train_lgbm(X_train, y_train, X_test, test_ids):
    print("\n═══ LightGBM (v3 — no balancing) ═══")
    le = LabelEncoder()
    y_enc = pd.Series(le.fit_transform(y_train), index=y_train.index)

    def objective(trial):
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
            "objective": "multiclass", "num_class": 3,
            "random_state": RANDOM_STATE, "n_jobs": -1, "verbosity": -1,
        }
        model = lgb.LGBMClassifier(**params)
        skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)
        scores = []
        for ti, vi in skf.split(X_train, y_enc):
            model.fit(X_train.iloc[ti], y_enc.iloc[ti])
            scores.append(accuracy_score(y_enc.iloc[vi], model.predict(X_train.iloc[vi])))
        return np.mean(scores)

    print("  Tuning...")
    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE))
    study.optimize(objective, n_trials=OPTUNA_TRIALS)
    bp = study.best_params
    bp.update({"objective": "multiclass", "num_class": 3, "random_state": RANDOM_STATE,
               "n_jobs": -1, "verbosity": -1})
    print(f"  Best: n_est={bp['n_estimators']}, depth={bp['max_depth']}, lr={bp['learning_rate']:.4f}")

    model = lgb.LGBMClassifier(**bp)
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    cv_scores = []
    for ti, vi in skf.split(X_train, y_enc):
        model.fit(X_train.iloc[ti], y_enc.iloc[ti])
        cv_scores.append(accuracy_score(y_enc.iloc[vi], model.predict(X_train.iloc[vi])))
    cv_scores = np.array(cv_scores)
    print(f"  CV: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

    ti, vi = list(skf.split(X_train, y_enc))[-1]
    model.fit(X_train.iloc[ti], y_enc.iloc[ti])
    report = classification_report(y_train.iloc[vi], le.inverse_transform(model.predict(X_train.iloc[vi])))

    model.fit(X_train, y_enc)
    importances = model.feature_importances_
    top_idx = np.argsort(importances)[::-1][:15]
    fi = "## Top 15 Feature Importances\n" + "\n".join(
        f"- {X_train.columns[i]}: {importances[i]:.0f}" for i in top_idx)

    save_submission(test_ids, le.inverse_transform(model.predict(X_test)), "lgbm")
    save_model_info("LightGBM", bp, cv_scores.mean(), cv_scores.std(), cv_scores, report, fi)
    return cv_scores.mean(), model, le


def train_ensemble(X_train, y_train, X_test, test_ids):
    """Soft voting ensemble of XGBoost + LightGBM + RF — no class balancing."""
    print("\n═══ Ensemble (v3 — Soft Voting, no balancing) ═══")
    le = LabelEncoder()
    y_enc = pd.Series(le.fit_transform(y_train), index=y_train.index)

    xgb_model = XGBClassifier(
        n_estimators=500, max_depth=7, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, min_child_weight=3,
        reg_alpha=0.1, reg_lambda=1.0,
        objective="multi:softprob", num_class=3,
        random_state=RANDOM_STATE, n_jobs=-1, verbosity=0)
    lgbm_model = lgb.LGBMClassifier(
        n_estimators=500, max_depth=8, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, min_child_samples=20,
        reg_alpha=0.1, reg_lambda=1.0, num_leaves=63,
        objective="multiclass", num_class=3,
        random_state=RANDOM_STATE, n_jobs=-1, verbosity=-1)
    rf_model = RandomForestClassifier(
        n_estimators=500, max_depth=20, min_samples_split=5,
        min_samples_leaf=2, max_features="sqrt",
        random_state=RANDOM_STATE, n_jobs=-1)

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    cv_scores = []

    for ti, vi in skf.split(X_train, y_enc):
        X_tr, X_val = X_train.iloc[ti], X_train.iloc[vi]
        y_tr_enc, y_val_enc = y_enc.iloc[ti], y_enc.iloc[vi]

        xgb_model.fit(X_tr, y_tr_enc)
        lgbm_model.fit(X_tr, y_tr_enc)
        rf_model.fit(X_tr, y_train.iloc[ti])

        xgb_probs = xgb_model.predict_proba(X_val)
        lgbm_probs = lgbm_model.predict_proba(X_val)
        rf_probs_raw = rf_model.predict_proba(X_val)
        rf_probs = np.zeros_like(rf_probs_raw)
        for i, cls in enumerate(rf_model.classes_):
            le_idx = np.where(le.classes_ == cls)[0][0]
            rf_probs[:, le_idx] = rf_probs_raw[:, i]

        avg_probs = (xgb_probs + lgbm_probs + rf_probs) / 3.0
        cv_scores.append(accuracy_score(y_val_enc, np.argmax(avg_probs, axis=1)))

    cv_scores = np.array(cv_scores)
    print(f"  CV: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

    # Last fold report
    ti, vi = list(skf.split(X_train, y_enc))[-1]
    xgb_model.fit(X_train.iloc[ti], y_enc.iloc[ti])
    lgbm_model.fit(X_train.iloc[ti], y_enc.iloc[ti])
    rf_model.fit(X_train.iloc[ti], y_train.iloc[ti])
    xp = xgb_model.predict_proba(X_train.iloc[vi])
    lp = lgbm_model.predict_proba(X_train.iloc[vi])
    rp_raw = rf_model.predict_proba(X_train.iloc[vi])
    rp = np.zeros_like(rp_raw)
    for i, cls in enumerate(rf_model.classes_):
        le_idx = np.where(le.classes_ == cls)[0][0]
        rp[:, le_idx] = rp_raw[:, i]
    avg = (xp + lp + rp) / 3.0
    report = classification_report(y_train.iloc[vi], le.inverse_transform(np.argmax(avg, axis=1)))

    # Full train
    xgb_model.fit(X_train, y_enc)
    lgbm_model.fit(X_train, y_enc)
    rf_model.fit(X_train, y_train)
    xp = xgb_model.predict_proba(X_test)
    lp = lgbm_model.predict_proba(X_test)
    rp_raw = rf_model.predict_proba(X_test)
    rp = np.zeros_like(rp_raw)
    for i, cls in enumerate(rf_model.classes_):
        le_idx = np.where(le.classes_ == cls)[0][0]
        rp[:, le_idx] = rp_raw[:, i]
    avg = (xp + lp + rp) / 3.0
    preds = le.inverse_transform(np.argmax(avg, axis=1))
    save_submission(test_ids, preds, "ensemble")

    save_model_info("Ensemble_SoftVoting", {
        "models": "XGBoost + LightGBM + RandomForest",
        "method": "Average predicted probabilities",
        "class_balancing": "None (accuracy-optimized)",
    }, cv_scores.mean(), cv_scores.std(), cv_scores, report)
    return cv_scores.mean()


def main():
    print("=" * 60)
    print("ACCURACY-OPTIMIZED MODEL PIPELINE (v3)")
    print("=" * 60)

    print("\nLoading data (unscaled)...")
    X_train, y_train, X_test, test_ids, _ = load_and_preprocess(TRAIN_PATH, TEST_PATH, scale=False)
    print(f"  Features: {X_train.shape[1]}, Train: {X_train.shape[0]}")

    print("\nLoading data (scaled for LogReg)...")
    X_train_s, y_train_s, X_test_s, test_ids_s, _ = load_and_preprocess(TRAIN_PATH, TEST_PATH, scale=True)

    results = {}
    results["LogReg_v3"] = train_logreg(X_train_s, y_train_s, X_test_s, test_ids_s)

    xgb_cv, _, _ = train_xgb(X_train, y_train, X_test, test_ids)
    results["XGB_v3"] = xgb_cv

    lgbm_cv, _, _ = train_lgbm(X_train, y_train, X_test, test_ids)
    results["LGBM_v3"] = lgbm_cv

    results["Ensemble_v3"] = train_ensemble(X_train, y_train, X_test, test_ids)

    print("\n" + "=" * 60)
    print("SUMMARY — 5-Fold Stratified CV Accuracy")
    print("=" * 60)
    for name, score in sorted(results.items(), key=lambda x: x[1], reverse=True):
        print(f"  {name:20s}: {score:.4f}")
    print("\nReference — v1 Kaggle scores: XGB=0.6804, LogReg=0.6745, RF=0.6703")
    print("Reference — v2 (balanced) CV: RF=0.6557, XGB=0.6511, LGBM=0.6462, LogReg=0.6325")


if __name__ == "__main__":
    main()

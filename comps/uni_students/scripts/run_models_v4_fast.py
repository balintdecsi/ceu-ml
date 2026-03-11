"""
Fast v4: Good defaults, no Optuna tuning. Uses v2 features.
XGBoost + LightGBM + Ensemble, 5-fold CV, ~1-2 min total.
"""
import pandas as pd, numpy as np, os, sys, warnings
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier
import lightgbm as lgb

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from preprocess_v2 import load_and_preprocess

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
SUBS = os.path.join(ROOT, "comps", "uni_students", "submissions")
INFO = os.path.join(ROOT, "comps", "uni_students", "models_info")
os.makedirs(SUBS, exist_ok=True)
os.makedirs(INFO, exist_ok=True)

def main():
    X, y, Xt, ids, _ = load_and_preprocess(
        os.path.join(ROOT, "data/uni_students/train.csv"),
        os.path.join(ROOT, "data/uni_students/test.csv"), scale=False)
    le = LabelEncoder()
    ye = pd.Series(le.fit_transform(y), index=y.index)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    models = {
        "xgb": XGBClassifier(n_estimators=600, max_depth=7, learning_rate=0.03,
            subsample=0.8, colsample_bytree=0.75, min_child_weight=5,
            reg_alpha=0.05, reg_lambda=0.2, objective="multi:softprob",
            num_class=3, random_state=42, n_jobs=-1, verbosity=0),
        "lgbm": lgb.LGBMClassifier(n_estimators=600, max_depth=7, learning_rate=0.03,
            subsample=0.8, colsample_bytree=0.75, min_child_samples=25,
            reg_alpha=0.05, reg_lambda=0.2, num_leaves=50,
            objective="multiclass", num_class=3, random_state=42, n_jobs=-1, verbosity=-1),
    }

    # CV + collect OOF probabilities for ensemble
    oof_probs = {k: np.zeros((len(X), 3)) for k in models}
    test_probs = {k: np.zeros((len(Xt), 3)) for k in models}
    cv_scores = {k: [] for k in models}

    for fold, (ti, vi) in enumerate(skf.split(X, ye)):
        for name, mdl in models.items():
            mdl.fit(X.iloc[ti], ye.iloc[ti])
            p = mdl.predict_proba(X.iloc[vi])
            oof_probs[name][vi] = p
            cv_scores[name].append(accuracy_score(ye.iloc[vi], np.argmax(p, axis=1)))
            test_probs[name] += mdl.predict_proba(Xt) / 5.0

    # Results
    for name in models:
        scores = cv_scores[name]
        mean, std = np.mean(scores), np.std(scores)
        print(f"{name:6s} CV: {mean:.4f} ± {std:.4f}  [{', '.join(f'{s:.4f}' for s in scores)}]")

        # Submission
        preds = le.inverse_transform(np.argmax(test_probs[name], axis=1))
        pd.DataFrame({"id": ids, "outcome": preds}).to_csv(
            os.path.join(SUBS, f"submission_{name}_v4.csv"), index=False)

        # Last fold report for info sheet
        ti, vi = list(skf.split(X, ye))[-1]
        report = classification_report(y.iloc[vi], le.inverse_transform(np.argmax(oof_probs[name][vi], axis=1)))
        with open(os.path.join(INFO, f"{name}_v4_info.md"), "w") as f:
            f.write(f"# {name} (v4 — fast defaults)\n\n")
            f.write(f"## CV: {mean:.4f} ± {std:.4f}\n\n")
            f.write(f"## Classification Report\n```\n{report}\n```\n")

    # Ensemble: average probabilities
    ens_test = sum(test_probs.values()) / len(test_probs)
    ens_oof = sum(oof_probs.values()) / len(oof_probs)
    ens_cv = []
    for _, vi in skf.split(X, ye):
        ens_cv.append(accuracy_score(ye.iloc[vi], np.argmax(ens_oof[vi], axis=1)))
    print(f"ensemb CV: {np.mean(ens_cv):.4f} ± {np.std(ens_cv):.4f}  [{', '.join(f'{s:.4f}' for s in ens_cv)}]")

    preds = le.inverse_transform(np.argmax(ens_test, axis=1))
    pd.DataFrame({"id": ids, "outcome": preds}).to_csv(
        os.path.join(SUBS, f"submission_ensemble_v4.csv"), index=False)

    report = classification_report(y, le.inverse_transform(np.argmax(ens_oof, axis=1)))
    with open(os.path.join(INFO, "ensemble_v4_info.md"), "w") as f:
        f.write(f"# Ensemble v4 (XGB+LGBM avg)\n\n")
        f.write(f"## CV: {np.mean(ens_cv):.4f} ± {np.std(ens_cv):.4f}\n\n")
        f.write(f"## OOF Classification Report\n```\n{report}\n```\n")

    print("\nDone! Submissions in comps/uni_students/submissions/")

if __name__ == "__main__":
    main()

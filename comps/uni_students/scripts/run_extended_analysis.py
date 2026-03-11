import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_predict
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from xgboost import XGBClassifier

from preprocess_utils import load_and_preprocess_data


RANDOM_STATE = 42
SCRIPT_DIR = Path(__file__).resolve().parent
COMP_DIR = SCRIPT_DIR.parent
REPO_ROOT = COMP_DIR.parent.parent


def _cv_folds_arg(value):
    folds = int(value)
    if folds < 2:
        raise argparse.ArgumentTypeError('cv-folds must be at least 2')
    return folds


def bootstrap_confidence_interval(scores, n_bootstrap=2000, random_state=RANDOM_STATE):
    rng = np.random.default_rng(random_state)
    scores = np.asarray(scores, dtype=float)
    samples = np.array([rng.choice(scores, size=len(scores), replace=True).mean() for _ in range(n_bootstrap)])
    return float(np.quantile(samples, 0.025)), float(np.quantile(samples, 0.975))


def _format_markdown_table(df):
    return df.to_markdown(index=True)


def _top_numeric_correlations(train_df, n=10):
    numeric = train_df.select_dtypes(include='number')
    if numeric.shape[1] < 2:
        return pd.DataFrame(columns=['feature_a', 'feature_b', 'abs_correlation'])

    corr = numeric.corr().abs()
    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
    pairs = (
        corr.where(mask)
        .stack()
        .sort_values(ascending=False)
        .head(n)
        .reset_index()
    )
    pairs.columns = ['feature_a', 'feature_b', 'abs_correlation']
    return pairs


def generate_extended_eda_report(train_path, output_path):
    train = pd.read_csv(train_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    module_outcomes = pd.crosstab(train['code_module'], train['outcome'], normalize='index').round(3)

    missing_score_cols = [col for col in ['mean_score', 'mean_score_TMA', 'mean_score_CMA'] if col in train.columns]
    missing_by_module = (
        train.groupby('code_module')[missing_score_cols].apply(lambda frame: frame.isna().mean()).round(3)
        if missing_score_cols
        else pd.DataFrame()
    )

    week_cols = [col for col in ['clicks_week1', 'clicks_week2', 'clicks_week3', 'clicks_week4'] if col in train.columns]
    click_trajectory = train.groupby('outcome')[week_cols].mean().round(1) if week_cols else pd.DataFrame()

    last_access_summary = (
        train.groupby('outcome')['last_access_day'].quantile([0.25, 0.5, 0.75]).unstack().round(1)
        if 'last_access_day' in train.columns
        else pd.DataFrame()
    )

    score_behaviour_cols = [col for col in ['mean_score', 'max_score', 'mean_days_late', 'num_assessments'] if col in train.columns]
    score_behaviour = train.groupby('outcome')[score_behaviour_cols].mean().round(2) if score_behaviour_cols else pd.DataFrame()

    correlations = _top_numeric_correlations(train.drop(columns=['id'], errors='ignore'))

    target_distribution = train['outcome'].value_counts(normalize=True).rename('proportion').round(3).to_frame()

    sections = [
        "# Extended EDA Summary",
        "",
        "## Scope",
        "- Focused on module heterogeneity, missingness, engagement trajectories, assessment behavior, and feature redundancy.",
        "- Chosen to directly address the bottlenecks documented in `reports/ANALYSIS_AND_IDEAS.md`: low Fail recall, structural missingness, and over-reliance on a single validation split.",
        "",
        "## Data Overview",
        f"- Training rows: {len(train):,}",
        f"- Columns: {train.shape[1]}",
        "",
        "### Outcome Distribution",
        _format_markdown_table(target_distribution),
        "",
        "### Outcome Mix by Module",
        _format_markdown_table(module_outcomes),
        "",
    ]

    if not missing_by_module.empty:
        sections.extend(
            [
                "### Assessment Missingness by Module",
                _format_markdown_table(missing_by_module),
                "",
            ]
        )

    if not click_trajectory.empty:
        sections.extend(
            [
                "### Mean Weekly Click Trajectory by Outcome",
                _format_markdown_table(click_trajectory),
                "",
            ]
        )

    if not last_access_summary.empty:
        sections.extend(
            [
                "### `last_access_day` Quartiles by Outcome",
                _format_markdown_table(last_access_summary),
                "",
            ]
        )

    if not score_behaviour.empty:
        sections.extend(
            [
                "### Mean Assessment / Submission Behavior by Outcome",
                _format_markdown_table(score_behaviour),
                "",
            ]
        )

    if not correlations.empty:
        sections.extend(
            [
                "### Strongest Numeric Correlations",
                correlations.to_markdown(index=False),
                "",
            ]
        )

    sections.extend(
        [
            "## Modeling Implications",
            "- Keep module identity explicit because both outcomes and assessment availability vary materially by module.",
            "- Preserve missingness as signal: all-score-missing rows need to distinguish structural no-assessment modules from behavioural non-submission.",
            "- Compare models with stratified cross-validation instead of a single holdout because the Fail class is the scarcest class.",
            "- Retain a neural-network baseline only after feature scaling, while tree models should use the unscaled engineered feature matrix.",
            "",
        ]
    )

    output_path.write_text("\n".join(sections), encoding='utf-8')


def _extract_fold_scores(search):
    best_index = search.best_index_
    split_keys = sorted(key for key in search.cv_results_ if key.startswith('split') and key.endswith('_test_score'))
    return [float(search.cv_results_[key][best_index]) for key in split_keys]


def _write_model_info(output_path, title, best_params, fold_scores, report, *, cv_folds, extra_lines=None):
    ci_low, ci_high = bootstrap_confidence_interval(fold_scores)
    mean_score = float(np.mean(fold_scores))
    std_score = float(np.std(fold_scores, ddof=1))

    lines = [
        f"# {title}",
        "",
        "## Validation Strategy",
        f"- {cv_folds}-fold stratified cross-validation",
        "- Bootstrap confidence interval computed from the fold-level accuracies",
        "",
        "## Best Hyperparameters",
    ]

    for key, value in best_params.items():
        lines.append(f"- {key}: {value}")

    lines.extend(
        [
            "",
            "## Cross-Validation Summary",
            f"- Fold accuracies: {', '.join(f'{score:.4f}' for score in fold_scores)}",
            f"- Mean accuracy: {mean_score:.4f}",
            f"- Std. accuracy: {std_score:.4f}",
            f"- Bootstrap 95% CI: [{ci_low:.4f}, {ci_high:.4f}]",
            "",
        ]
    )

    if extra_lines:
        lines.extend(extra_lines)
        lines.append("")

    lines.extend(
        [
            "## Out-of-Fold Classification Report",
            "```text",
            report,
            "```",
        ]
    )

    Path(output_path).write_text("\n".join(lines), encoding='utf-8')


def run_model_suite(
    train_path,
    test_path,
    models_info_dir,
    submissions_dir,
    *,
    include_mlp=False,
    cv_folds=3,
):
    models_info_dir = Path(models_info_dir)
    submissions_dir = Path(submissions_dir)
    models_info_dir.mkdir(parents=True, exist_ok=True)
    submissions_dir.mkdir(parents=True, exist_ok=True)

    X_tree, y_train, X_test_tree, test_ids = load_and_preprocess_data(
        train_path,
        test_path,
        scale_numeric=False,
        add_engineered_features=True,
    )

    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=RANDOM_STATE)
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y_train)

    xgb_start = time.perf_counter()
    xgb_search = GridSearchCV(
        estimator=XGBClassifier(
            objective='multi:softprob',
            num_class=len(label_encoder.classes_),
            eval_metric='mlogloss',
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
        param_grid={
            # Keep the grid intentionally small so the default submission workflow stays under
            # the time budget: tree depth and child weight are searched, while tree count,
            # learning rate, subsampling, and column subsampling stay fixed to sensible defaults.
            'n_estimators': [250],
            'max_depth': [4, 6],
            'learning_rate': [0.05],
            'subsample': [0.8],
            'colsample_bytree': [0.8],
            'min_child_weight': [1, 3],
        },
        scoring='accuracy',
        cv=cv,
        n_jobs=1,
        refit=True,
    )
    xgb_search.fit(X_tree, y_encoded)
    best_xgb = xgb_search.best_estimator_
    xgb_fold_scores = _extract_fold_scores(xgb_search)
    xgb_oof = label_encoder.inverse_transform(cross_val_predict(best_xgb, X_tree, y_encoded, cv=cv, method='predict'))
    xgb_report = classification_report(y_train, xgb_oof)

    feature_importances = best_xgb.feature_importances_
    feature_order = np.argsort(feature_importances)[::-1][:10]
    top_features = [
        "## Top Feature Importances",
        *[
            f"- {X_tree.columns[idx]}: {feature_importances[idx]:.4f}"
            for idx in feature_order
        ],
    ]

    _write_model_info(
        models_info_dir / 'TunedXGBoost_info.md',
        'Tuned XGBoost',
        xgb_search.best_params_,
        xgb_fold_scores,
        xgb_report,
        cv_folds=cv_folds,
        extra_lines=top_features,
    )

    xgb_test_predictions = label_encoder.inverse_transform(best_xgb.predict(X_test_tree))
    pd.DataFrame({'id': test_ids, 'outcome': xgb_test_predictions}).to_csv(
        submissions_dir / 'submission_xgb_tuned.csv',
        index=False,
    )
    print(f"Tuned XGBoost search + fit completed in {time.perf_counter() - xgb_start:.1f}s")

    if include_mlp:
        mlp_start = time.perf_counter()
        mlp_search = GridSearchCV(
            estimator=Pipeline(
                [
                    ('scaler', StandardScaler()),
                    (
                        'mlp',
                        MLPClassifier(
                            max_iter=250,
                            early_stopping=True,
                            random_state=RANDOM_STATE,
                        ),
                    ),
                ]
            ),
            param_grid={
                'mlp__hidden_layer_sizes': [(128, 64), (64, 32)],
                'mlp__alpha': [1e-4],
                'mlp__learning_rate_init': [1e-3],
            },
            scoring='accuracy',
            cv=cv,
            n_jobs=1,
            refit=True,
        )
        mlp_search.fit(X_tree, y_encoded)
        best_mlp = mlp_search.best_estimator_
        mlp_fold_scores = _extract_fold_scores(mlp_search)
        mlp_oof = label_encoder.inverse_transform(cross_val_predict(best_mlp, X_tree, y_encoded, cv=cv, method='predict'))
        mlp_report = classification_report(y_train, mlp_oof)

        _write_model_info(
            models_info_dir / 'NeuralNetwork_info.md',
            'Neural Network (MLPClassifier)',
            mlp_search.best_params_,
            mlp_fold_scores,
            mlp_report,
            cv_folds=cv_folds,
            extra_lines=[
                "## Architecture Notes",
                "- Input features come from the engineered preprocessing pipeline.",
                "- StandardScaler is applied inside the sklearn pipeline before the MLP.",
                "- Early stopping is enabled to limit overfitting on the tabular feature matrix.",
            ],
        )

        mlp_test_predictions = label_encoder.inverse_transform(best_mlp.predict(X_test_tree))
        pd.DataFrame({'id': test_ids, 'outcome': mlp_test_predictions}).to_csv(
            submissions_dir / 'submission_mlp.csv',
            index=False,
        )
        print(f"MLP search + fit completed in {time.perf_counter() - mlp_start:.1f}s")


def parse_args():
    parser = argparse.ArgumentParser(description='Run the extended EDA and tuned model workflow for the uni_students competition.')
    parser.add_argument('--train-path', default=str(REPO_ROOT / 'data/uni_students/train.csv'))
    parser.add_argument('--test-path', default=str(REPO_ROOT / 'data/uni_students/test.csv'))
    parser.add_argument('--eda-output', default=str(COMP_DIR / 'reports/EXTENDED_EDA_REPORT.md'))
    parser.add_argument('--models-info-dir', default=str(COMP_DIR / 'models_info'))
    parser.add_argument('--submissions-dir', default=str(COMP_DIR / 'submissions'))
    parser.add_argument('--include-mlp', action='store_true', help='Also train the MLP baseline. Disabled by default to keep the run fast.')
    parser.add_argument('--cv-folds', type=_cv_folds_arg, default=3, help='Number of stratified CV folds. Defaults to 3 for a sub-15-minute run.')
    return parser.parse_args()


def main():
    args = parse_args()
    generate_extended_eda_report(args.train_path, args.eda_output)
    run_model_suite(
        args.train_path,
        args.test_path,
        args.models_info_dir,
        args.submissions_dir,
        include_mlp=args.include_mlp,
        cv_folds=args.cv_folds,
    )


if __name__ == '__main__':
    main()

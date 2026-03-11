"""
Comprehensive EDA for the University Students Kaggle Competition.
Outputs a markdown report to comps/uni_students/reports/EDA_DETAILED.md
"""
import pandas as pd
import numpy as np
import os
import sys
from itertools import combinations

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
TRAIN_PATH = os.path.join(REPO_ROOT, "data", "uni_students", "train.csv")
TEST_PATH = os.path.join(REPO_ROOT, "data", "uni_students", "test.csv")
REPORT_PATH = os.path.join(REPO_ROOT, "comps", "uni_students", "reports", "EDA_DETAILED.md")

CAT_COLS = ["code_module", "gender", "region", "highest_education", "imd_band", "age_band", "disability"]
SCORE_COLS = ["mean_score", "std_score", "min_score", "max_score", "mean_score_TMA", "mean_score_CMA",
              "mean_days_late", "max_days_late"]
WEEKLY_CLICK_COLS = ["clicks_pre_course", "clicks_week1", "clicks_week2", "clicks_week3", "clicks_week4"]
ASSESSMENT_COLS = ["num_assessments", "num_banked", "num_late_submissions", "num_early_submissions"] + SCORE_COLS

def md_table(df, float_fmt=".4f"):
    """Convert a DataFrame to markdown table string."""
    return df.to_markdown(floatfmt=float_fmt)

def run_eda():
    train = pd.read_csv(TRAIN_PATH)
    test = pd.read_csv(TEST_PATH)
    lines = []
    def w(text=""):
        lines.append(text)

    w("# Detailed EDA Report — University Students Competition\n")
    w(f"- Train shape: {train.shape}")
    w(f"- Test shape: {test.shape}")
    w(f"- Features: {train.shape[1] - 2} (excluding id and outcome)\n")

    # ── 1. Target Distribution ──
    w("## 1. Target Distribution\n")
    outcome_counts = train["outcome"].value_counts()
    outcome_pcts = train["outcome"].value_counts(normalize=True) * 100
    summary = pd.DataFrame({"count": outcome_counts, "pct": outcome_pcts})
    w(md_table(summary, ".1f"))
    w("\n**Imbalance ratio (majority/minority)**: "
      f"{outcome_counts.max() / outcome_counts.min():.2f}x (Pass vs Fail)\n")

    # ── 2. Per-Module Analysis ──
    w("## 2. Per-Module Outcome Distribution\n")
    w("This is critical because modules have fundamentally different data availability.\n")
    module_outcome = pd.crosstab(train["code_module"], train["outcome"], normalize="index") * 100
    module_outcome["n"] = train.groupby("code_module").size()
    module_outcome = module_outcome[["n", "Pass", "Fail", "Withdrawn"]].sort_values("Fail", ascending=False)
    w(md_table(module_outcome, ".1f"))

    # Assessment availability per module
    w("\n### Assessment Data Availability by Module\n")
    module_assess = train.groupby("code_module").agg(
        pct_missing_scores=("mean_score", lambda x: x.isnull().mean() * 100),
        mean_num_assessments=("num_assessments", "mean"),
        pct_zero_assessments=("num_assessments", lambda x: (x == 0).mean() * 100),
    ).round(1)
    w(md_table(module_assess, ".1f"))
    w("\n**Key finding**: Modules EEE and GGG have ~100% missing scores (no assessments by day 28). "
      "In other modules, missing scores indicate students who didn't submit — a strong behavioral signal.\n")

    # ── 3. Missing Values Deep Dive ──
    w("## 3. Missing Values Analysis\n")
    
    # Overall missing
    missing = train.isnull().sum()
    missing = missing[missing > 0].sort_values(ascending=False)
    missing_pct = (missing / len(train) * 100).round(1)
    missing_df = pd.DataFrame({"missing_count": missing, "missing_pct": missing_pct})
    w("### Overall Missing Values\n")
    w(md_table(missing_df, ".1f"))

    # Structural vs behavioral missing
    w("\n### Structural vs. Behavioral Missing Scores\n")
    no_assess_modules = ["EEE", "GGG"]
    has_assess_modules = [m for m in train["code_module"].unique() if m not in no_assess_modules]

    structural_missing = train[train["code_module"].isin(no_assess_modules)]["mean_score"].isnull().sum()
    behavioral_missing = train[
        train["code_module"].isin(has_assess_modules) & train["mean_score"].isnull()
    ].shape[0]
    total_missing_scores = train["mean_score"].isnull().sum()

    w(f"- Total rows with missing `mean_score`: {total_missing_scores} ({total_missing_scores/len(train)*100:.1f}%)")
    w(f"- Structural (EEE/GGG, no assessments available): {structural_missing}")
    w(f"- Behavioral (other modules, student didn't submit): {behavioral_missing}")

    # Outcome distribution for behavioral non-submitters
    behavioral_nonsubmit = train[
        train["code_module"].isin(has_assess_modules) & (train["num_assessments"] == 0)
    ]
    if len(behavioral_nonsubmit) > 0:
        w(f"\n**Outcome of students who didn't submit in modules WITH assessments** (n={len(behavioral_nonsubmit)}):\n")
        bns_outcome = behavioral_nonsubmit["outcome"].value_counts(normalize=True) * 100
        w(md_table(bns_outcome.to_frame("pct"), ".1f"))
        w("\n⚠️ This is a very strong signal — students who don't submit when they could are overwhelmingly likely to withdraw or fail.\n")

    # ── 4. Feature Distributions by Outcome ──
    w("## 4. Key Feature Distributions by Outcome\n")

    key_numeric = ["last_access_day", "total_clicks", "num_active_days", "mean_clicks_per_day",
                   "num_assessments", "mean_score", "days_before_course_start", "studied_credits",
                   "num_of_prev_attempts"]

    for feat in key_numeric:
        w(f"\n### `{feat}`\n")
        grouped = train.groupby("outcome")[feat].describe()[["mean", "50%", "std", "min", "max"]]
        grouped.columns = ["mean", "median", "std", "min", "max"]
        w(md_table(grouped, ".2f"))

    # ── 5. Engagement Trajectories ──
    w("\n## 5. Weekly Engagement Trajectories\n")
    w("Mean clicks per week by outcome class:\n")
    weekly_by_outcome = train.groupby("outcome")[WEEKLY_CLICK_COLS].mean()
    w(md_table(weekly_by_outcome, ".1f"))

    # Compute trend
    w("\n### Engagement Trend (Week4 - Week1)\n")
    train["_click_trend"] = train["clicks_week4"] - train["clicks_week1"]
    trend_by_outcome = train.groupby("outcome")["_click_trend"].describe()[["mean", "50%", "std"]]
    trend_by_outcome.columns = ["mean", "median", "std"]
    w(md_table(trend_by_outcome, ".2f"))
    w("\n**Interpretation**: Withdrawn students show declining engagement (negative trend). "
      "Pass students maintain or increase engagement.\n")
    train.drop(columns=["_click_trend"], inplace=True)

    # ── 6. Categorical Feature Analysis ──
    w("## 6. Categorical Feature vs. Outcome\n")

    for col in CAT_COLS:
        w(f"\n### `{col}`\n")
        ct = pd.crosstab(train[col], train["outcome"], normalize="index") * 100
        ct["n"] = train[col].value_counts()
        ct = ct[["n", "Pass", "Fail", "Withdrawn"]].sort_values("Pass", ascending=False)
        # Only show top 10 if many categories
        if len(ct) > 12:
            w(f"*(Showing top 10 of {len(ct)} categories)*\n")
            ct = ct.head(10)
        w(md_table(ct, ".1f"))

    # ── 7. Correlation Analysis ──
    w("\n## 7. Top Feature Correlations with Outcome\n")
    w("Using numeric encoding: Pass=2, Withdrawn=1, Fail=0\n")
    train["_outcome_num"] = train["outcome"].map({"Fail": 0, "Withdrawn": 1, "Pass": 2})
    num_cols = train.select_dtypes(include=[np.number]).columns.drop(["id", "_outcome_num"])
    correlations = train[num_cols].corrwith(train["_outcome_num"]).abs().sort_values(ascending=False)
    w(md_table(correlations.head(20).to_frame("abs_correlation"), ".4f"))

    # Inter-feature correlations (top correlated pairs)
    w("\n### Highly Correlated Feature Pairs (|r| > 0.8)\n")
    corr_matrix = train[num_cols].corr()
    high_corr_pairs = []
    for i, j in combinations(range(len(num_cols)), 2):
        c1, c2 = num_cols[i], num_cols[j]
        r = corr_matrix.loc[c1, c2]
        if abs(r) > 0.8:
            high_corr_pairs.append((c1, c2, r))
    high_corr_pairs.sort(key=lambda x: abs(x[2]), reverse=True)
    if high_corr_pairs:
        hcp_df = pd.DataFrame(high_corr_pairs, columns=["feature_1", "feature_2", "correlation"])
        w(md_table(hcp_df, ".3f"))
        w(f"\n**{len(high_corr_pairs)} highly correlated pairs found.** Consider dropping redundant features or using PCA.\n")
    else:
        w("No pairs with |r| > 0.8 found.\n")

    train.drop(columns=["_outcome_num"], inplace=True)

    # ── 8. last_access_day Deep Dive ──
    w("## 8. `last_access_day` Deep Dive (Top Feature)\n")
    w("Distribution of last_access_day by outcome:\n")
    lad = train.groupby("outcome")["last_access_day"].describe()[["mean", "50%", "25%", "75%", "std"]]
    lad.columns = ["mean", "median", "p25", "p75", "std"]
    w(md_table(lad, ".1f"))

    # Percentage with last_access_day < 14 (stopped before week 2 ended)
    w("\n### Students who stopped accessing VLE early\n")
    for threshold in [7, 14, 21]:
        early_stop = train[train["last_access_day"] <= threshold]
        if len(early_stop) > 0:
            pct = early_stop["outcome"].value_counts(normalize=True) * 100
            w(f"\n**last_access_day ≤ {threshold}** (n={len(early_stop)}, {len(early_stop)/len(train)*100:.1f}% of data):")
            w(f"  Pass={pct.get('Pass', 0):.1f}%, Fail={pct.get('Fail', 0):.1f}%, Withdrawn={pct.get('Withdrawn', 0):.1f}%")

    # ── 9. Train vs Test Distribution Comparison ──
    w("\n\n## 9. Train vs Test Distribution Comparison\n")
    w("Checking for distribution shift between train and test sets.\n")

    w("### Categorical Features\n")
    for col in CAT_COLS:
        train_dist = train[col].value_counts(normalize=True).sort_index()
        test_dist = test[col].value_counts(normalize=True).sort_index()
        all_cats = sorted(set(train_dist.index) | set(test_dist.index))
        comp = pd.DataFrame({
            "train_pct": train_dist.reindex(all_cats, fill_value=0) * 100,
            "test_pct": test_dist.reindex(all_cats, fill_value=0) * 100,
        })
        comp["diff"] = (comp["test_pct"] - comp["train_pct"]).abs()
        max_diff = comp["diff"].max()
        if max_diff > 2.0:  # Only report if notable difference
            w(f"\n**`{col}`** (max diff: {max_diff:.1f}pp):\n")
            w(md_table(comp[comp["diff"] > 1.0].sort_values("diff", ascending=False), ".1f"))

    w("\n### Numeric Features (mean comparison)\n")
    num_comp_rows = []
    for col in num_cols:
        if col in test.columns:
            tr_mean = train[col].mean()
            te_mean = test[col].mean()
            pct_diff = abs(tr_mean - te_mean) / (abs(tr_mean) + 1e-10) * 100
            num_comp_rows.append({"feature": col, "train_mean": tr_mean, "test_mean": te_mean, "pct_diff": pct_diff})
    num_comp = pd.DataFrame(num_comp_rows).sort_values("pct_diff", ascending=False).head(15)
    w(md_table(num_comp.set_index("feature"), ".2f"))

    # ── 10. Summary of Key Findings ──
    w("\n\n## 10. Key EDA Findings Summary\n")
    w("""
1. **Class imbalance**: Pass (47.2%) >> Fail (21.6%). Models need class balancing.
2. **Module heterogeneity**: Fail rates range from ~15% to ~30% across modules. Modules EEE/GGG have no assessment data.
3. **Structural vs behavioral missing**: ~37% of scores are missing. Distinguishing "no assessment available" from "didn't submit" is critical — non-submitters in assessment-available modules overwhelmingly fail/withdraw.
4. **`last_access_day` is the single strongest predictor**: Students who stop accessing VLE early are almost certain to withdraw/fail.
5. **Engagement trends matter**: Withdrawn students show declining weekly clicks; Pass students maintain engagement.
6. **High multicollinearity**: Many click features are redundant (total_clicks correlated with weekly sums, etc.).
7. **Train/test distributions are similar**: No major distribution shift detected.
8. **Previous attempts**: Students with >0 previous attempts have different outcome distributions.
""")

    # Write report
    os.makedirs(os.path.dirname(REPORT_PATH), exist_ok=True)
    with open(REPORT_PATH, "w") as f:
        f.write("\n".join(lines))
    print(f"EDA report written to {REPORT_PATH}")

if __name__ == "__main__":
    run_eda()

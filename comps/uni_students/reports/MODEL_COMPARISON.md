# Model Comparison & Results Summary

## Kaggle Competition: University Student Outcome Prediction

### Overview
- **Task**: 3-class classification (Pass / Fail / Withdrawn) from early-semester data
- **Metric**: Accuracy
- **Baseline**: Always predict "Pass" → 47.2%

---

## All Model Results

### v1 — Original Models (single 80/20 split, no feature engineering)

| Model               | Val Accuracy | Kaggle Test |
|---------------------|-------------|-------------|
| XGBoost             | 66.58%      | **68.04%**  |
| Logistic Regression | 65.86%      | 67.45%      |
| Random Forest       | 65.53%      | 67.03%      |

### v2 — Class-Balanced + Feature Engineering (5-fold Stratified CV)

| Model               | CV Accuracy       |
|---------------------|-------------------|
| Random Forest       | 65.57% ± 0.42%   |
| Ensemble (Soft)     | 65.41% ± 0.52%   |
| XGBoost             | 65.11% ± 0.68%   |
| LightGBM            | 64.62% ± 0.49%   |
| Logistic Regression | 63.25% ± 0.70%   |

**Observation**: Class balancing *hurts* accuracy. By redistributing predictions from Pass to Fail, we lose more correct Pass predictions than we gain correct Fail predictions.

### v3 — Accuracy-Optimized + Feature Engineering (5-fold Stratified CV, Optuna-tuned)

| Model               | CV Accuracy       |
|---------------------|-------------------|
| **XGBoost**         | **66.60% ± 0.32%** |
| LightGBM            | 66.48% ± 0.58%   |
| Ensemble (Soft)     | 66.46% ± 0.44%   |
| Logistic Regression | 66.12% ± 0.20%   |

---

## Key Improvements in v2/v3 Pipeline

### Feature Engineering (20 new features)
1. **`chose_not_to_submit`** — students in modules WITH assessments who submitted nothing (→ became the **#1 most important XGBoost feature** at 0.246 importance)
2. **`no_assess_available`** — structural missing data flag (modules EEE/GGG)
3. **`stopped_early`** — student stopped accessing VLE before day 14
4. **`click_trend_w4_w1`** — engagement trajectory (week 4 - week 1 clicks)
5. **`early_vs_late_ratio`** — early engagement relative to late
6. **`vle_access_span`** — days between first and last VLE access
7. **`active_day_ratio`** — proportion of 28 days actively used
8. Missing indicators for all score columns
9. Various behavioral flags (all_early_submissions, any_late, registered_late, etc.)

### Methodology Improvements
- **5-fold Stratified CV** instead of single 80/20 split
- **Optuna hyperparameter tuning** (15 trials × 3-fold inner CV per model)
- **Proper train/test encoding** — one-hot encoding fit on train only (no test leakage)
- **No scaling for tree models** — only LogReg gets StandardScaler
- **LightGBM** added as new model type

---

## Critical EDA Findings

1. **Behavioral non-submission is the strongest signal**: Students who *could* submit but didn't (in modules with assessments) have a 96% fail/withdraw rate. The `chose_not_to_submit` feature alone has 2× the importance of any other feature in XGBoost.

2. **`last_access_day` remains the #2 predictor**: Students who stop accessing the VLE early almost certainly withdraw. At `last_access_day ≤ 7`, withdrawal rate exceeds 85%.

3. **Module heterogeneity is extreme**: Pass rates range from 38% (CCC) to 71% (AAA). Module GGG has 29% fail rate vs. AAA's 12%.

4. **Class balancing hurts accuracy**: v2 (balanced) scored ~1% lower than v3 (unbalanced) across all models. The accuracy metric rewards correctly predicting the majority class.

5. **High multicollinearity among click features**: Many click features are redundant (total_clicks ≈ sum of weekly clicks ≈ sum of activity clicks), but tree models handle this gracefully.

---

## Submission Files Generated

| File | Model | Notes |
|------|-------|-------|
| `submission_xgb_v3.csv` | XGBoost (best) | Best CV, recommended for submission |
| `submission_lgbm_v3.csv` | LightGBM | Close second |
| `submission_ensemble_v3.csv` | Ensemble | XGB + LGBM + RF soft voting |
| `submission_logreg_v3.csv` | LogReg | Best linear model |
| `submission_*_v2.csv` | v2 variants | Class-balanced versions |

---

## Files Created / Modified

### Scripts
- `scripts/preprocess_v2.py` — Improved preprocessing with feature engineering
- `scripts/run_eda_detailed.py` — Comprehensive EDA generating `EDA_DETAILED.md`
- `scripts/run_models_v2.py` — Class-balanced models with Optuna tuning
- `scripts/run_models_v3.py` — Accuracy-optimized models with Optuna tuning

### Reports
- `reports/EDA_DETAILED.md` — Full EDA with 10 sections (target, per-module, missing values, feature distributions, trajectories, categoricals, correlations, last_access_day deep dive, train/test comparison, summary)
- `reports/ANALYSIS_AND_IDEAS.md` — Initial analysis and brainstorming
- `reports/MODEL_COMPARISON.md` — This file

### Model Info
- `models_info/*_v2_info.md` — v2 model cards (5 files)
- `models_info/*_v3_info.md` — v3 model cards (4 files)

### Submissions
- `submissions/submission_*_v2.csv` — v2 predictions (5 files)
- `submissions/submission_*_v3.csv` — v3 predictions (4 files)

---

## Further Improvement Ideas (Not Yet Attempted)

1. **Per-module models**: Train separate models for EEE/GGG (no assessment features) vs. other modules
2. **Target encoding** for categorical features instead of one-hot
3. **More Optuna trials**: 15 trials is minimal; 50-100 would likely find better hyperparameters
4. **Early stopping** for gradient boosting models to prevent overfitting
5. **Stacking**: Use out-of-fold predictions as features for a meta-learner
6. **Two-stage classification**: Pass vs Not-Pass, then Fail vs Withdrawn

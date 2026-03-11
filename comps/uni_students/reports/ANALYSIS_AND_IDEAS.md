# Analysis of Current Models & Ideas for Improvement

## 1. Current State Summary

### Competition
- **Task**: Predict student outcome (Pass / Withdrawn / Fail) using early-semester data (first 4 weeks)
- **Metric**: Accuracy
- **Baseline**: Always predict "Pass" → 47.2% accuracy
- **Data**: 24,444 train rows, 8,149 test rows, 53 features

### Models Trained

| Model               | Val Accuracy | Fail Recall | Pass Recall | Withdrawn Recall |
|---------------------|-------------|-------------|-------------|------------------|
| Logistic Regression | 65.86%      | 19%         | 90%         | 62%              |
| Random Forest       | 65.53%      | 14%         | 92%         | 61%              |
| XGBoost             | 66.58%      | 21%         | 90%         | 62%              |

**Key observation**: All models heavily over-predict Pass and severely under-predict Fail (recall 14–21%). This is the main bottleneck — correctly classifying even a few more Fail students would meaningfully boost accuracy.

### Top Features (from XGBoost)
1. `last_access_day` (0.122) — strongest signal by far
2. `code_module_GGG` (0.077)
3. `num_assessments` (0.057)
4. `max_score` (0.048)
5. `mean_score_TMA_is_missing` (0.043)

---

## 2. Issues Found in Current Pipeline

### 2.1 Submission Column Name Inconsistency
- LogReg and RF submissions use `ID,outcome` as header
- XGBoost submission uses `id,outcome` as header
- The Kaggle overview example shows `ID,outcome` — verify which one Kaggle actually accepts to avoid silent scoring failures.

### 2.2 Preprocessing Concerns

**Log transform applied after median imputation**: In `preprocess_utils.py`, missing values are first imputed with median, and *then* log transforms are applied. This means imputed values get log-transformed along with real values. Since median imputation already inserts a "neutral" value, the log transform on those imputed values may not add useful signal and could introduce noise.

**StandardScaler on everything**: The scaler is applied to all columns including one-hot encoded features and missing indicator flags (0/1 binary). Scaling binary dummies changes their interpretation and can hurt tree-based models (RF, XGBoost) that rely on split thresholds. Tree models generally don't benefit from scaling at all.

**No differentiation between "no assessment available" vs. "didn't submit"**: The missing value indicators are binary, but there's a critical distinction:
- Students in modules EEE/GGG have *structurally* missing assessment data (no assessments due by day 28)
- Students in other modules who have missing scores *chose not to submit* — a very different signal

Both cases currently get the same missing indicator + median imputation.

### 2.3 Validation Strategy
All models use a single random 80/20 split with `random_state=42`. This gives a noisy accuracy estimate and risks overfitting to one particular split. Stratified K-fold cross-validation would be more reliable.

### 2.4 No Hyperparameter Tuning
All models use default or near-default hyperparameters. The XGBoost model uses only 100 trees with `max_depth=6` and `learning_rate=0.1` — likely underfitting.

---

## 3. EDA Ideas

### 3.1 Per-Module Analysis (High Priority)
The data description emphasizes that modules vary dramatically. Key questions:
- What's the outcome distribution *per module*? Some modules might be much harder.
- Which modules have assessments by day 28 vs. not? (AAA/BBB/DDD have TMAs; EEE/GGG have none)
- Do VLE engagement patterns differ by module?
- **Idea**: Plot outcome rates per module as a grouped bar chart. This could motivate per-module models or module-specific feature engineering.

### 3.2 Missing Data Pattern Analysis (High Priority)
- Cross-tabulate missingness of assessment scores with `code_module` to confirm the structural vs. behavioral distinction
- For modules *with* assessments available, what % of Fail/Withdrawn students have missing scores vs. Pass students? (Hypothesis: non-submission strongly predicts failure)
- Create a heatmap of missing patterns across all features

### 3.3 Engagement Trajectory Analysis
- Plot weekly click trends (clicks_week1 → clicks_week4) by outcome class
- Compute week-over-week engagement change (e.g., `clicks_week4 - clicks_week1` or ratio `clicks_week4 / clicks_week1`)
- Hypothesis: declining engagement predicts Withdrawal; low but steady engagement predicts Fail
- Plot `last_access_day` distribution by outcome — this is already the #1 feature, worth visualizing

### 3.4 Assessment Score Distributions by Outcome
- For students who *have* scores: how do score distributions differ between Pass/Fail/Withdrawn?
- Scatter plot: `mean_score` vs. `mean_days_late` colored by outcome
- Are there score thresholds that cleanly separate classes?

### 3.5 Feature Correlation Analysis
- Correlation matrix of numeric features — many click features are likely highly correlated (total_clicks, weekly clicks, activity clicks)
- VIF (Variance Inflation Factor) analysis for multicollinearity
- This matters especially for Logistic Regression performance

### 3.6 Class Separability
- PCA or t-SNE visualization of the feature space colored by outcome
- This would reveal whether classes are fundamentally separable or heavily overlapping
- If Fail and Withdrawn overlap heavily, a 2-class model (Pass vs. Not Pass) might perform better before splitting Not Pass further

### 3.7 Registration Timing
- `days_before_course_start` vs. outcome — do late registrants fare worse?
- `first_access_day` vs. outcome — does early VLE engagement predict success?

---

## 4. Feature Engineering Ideas

### 4.1 Engagement Trend Features
```
click_trend = clicks_week4 - clicks_week1
click_ratio_w4_w1 = clicks_week4 / (clicks_week1 + 1)
engagement_consistency = std([clicks_week1, clicks_week2, clicks_week3, clicks_week4])
active_day_ratio = num_active_days / 28
```

### 4.2 Smart Missing Value Features
```
has_no_assessments_available = code_module in ['EEE', 'GGG']  (structural)
chose_not_to_submit = (num_assessments == 0) & (code_module not in ['EEE', 'GGG'])  (behavioral)
```
This distinction could be very powerful. A student who *could* submit but *didn't* is very different from one who had nothing to submit.

### 4.3 Interaction Features
```
score_x_engagement = mean_score * total_clicks
early_engagement = clicks_pre_course + clicks_week1
late_engagement = clicks_week3 + clicks_week4
early_vs_late = early_engagement / (late_engagement + 1)
```

### 4.4 Per-Module Normalized Features
Normalize VLE engagement features *within each module* (z-score within module group). A student with 200 clicks in module AAA and 200 clicks in module GGG may have very different relative engagement levels.

### 4.5 Assessment Behavior Features
```
submission_rate = num_assessments / expected_assessments_per_module
all_early = (num_early_submissions == num_assessments)
any_late = (num_late_submissions > 0)
```

---

## 5. Model Improvement Ideas

### 5.1 Quick Wins
- **Class weights**: All three sklearn/xgboost models support `class_weight='balanced'` or `scale_pos_weight`. This should immediately improve Fail recall.
- **Stratified K-Fold CV**: Replace single split with 5-fold stratified CV for more robust evaluation.
- **XGBoost hyperparameter tuning**: Increase `n_estimators` to 500+, reduce `learning_rate` to 0.01–0.05, tune `max_depth`, `subsample`, `colsample_bytree`, `min_child_weight`, `reg_alpha`, `reg_lambda`.
- **Remove scaling for tree models**: Don't StandardScale data fed to RF/XGBoost — it's unnecessary and the log transforms already reduce skew.

### 5.2 Additional Models to Try
- **LightGBM**: Often faster and competitive with XGBoost
- **CatBoost**: Handles categorical features natively (no one-hot encoding needed), often strong out-of-the-box
- **Stacking/Blending**: Combine LogReg + RF + XGBoost predictions as inputs to a meta-learner
- **Neural Network**: Simple MLP with dropout — the class notebooks include deep learning intro

### 5.3 Per-Module Models
Train separate models for modules with/without assessment data. The feature space is fundamentally different:
- Modules EEE/GGG: Only demographic + VLE features available
- Other modules: Full feature set including assessment scores

A single model must learn this bifurcation implicitly; separate models can specialize.

### 5.4 Two-Stage Classification
1. First predict **Pass vs. Not Pass** (binary, easier)
2. Then for predicted Not Pass, predict **Fail vs. Withdrawn**

This hierarchical approach may help because Pass is the most separable class.

### 5.5 Threshold Tuning
For models that output probabilities, tune the decision thresholds per class rather than using argmax. This can help allocate more predictions to the underrepresented Fail class.

### 5.6 Optuna/Bayesian Hyperparameter Optimization
Use Optuna for automated hyperparameter search with cross-validation, targeting accuracy directly.

---

## 6. Priority Recommendations

| Priority | Action | Expected Impact |
|----------|--------|-----------------|
| 🔴 High | Add class weights to existing models | Immediate Fail recall boost |
| 🔴 High | Smart missing value encoding (structural vs. behavioral) | Better signal for ~37% of data |
| 🔴 High | Stratified K-Fold CV | More reliable model comparison |
| 🟡 Medium | Engagement trend features | New predictive signal |
| 🟡 Medium | XGBoost hyperparameter tuning (Optuna) | 2–5% accuracy gain likely |
| 🟡 Medium | Try CatBoost | Strong baseline with less preprocessing |
| 🟡 Medium | Per-module EDA + per-module models | Better handling of heterogeneous data |
| 🟢 Low | Stacking ensemble | Marginal gains on top of tuned models |
| 🟢 Low | Two-stage classification | Worth testing if Fail remains hard |
| 🟢 Low | Neural network | Diminishing returns for tabular data |

---

## 7. Minor Issues to Fix
- **Submission header inconsistency**: LogReg/RF use `ID` while XGBoost uses `id`. Standardize to match what Kaggle expects.
- **OLS submission goes to `data/uni_students/`** instead of `comps/uni_students/submissions/` like the other models.
- **Preprocessing leaks test info**: `preprocess_utils.py` concatenates train+test for one-hot encoding (`pd.concat([X_train, X_test])`). While this ensures column alignment, it means the encoding is influenced by test set category distributions. Use `fit` on train only and `transform` on test for stricter separation.

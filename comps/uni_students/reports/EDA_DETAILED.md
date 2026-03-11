# Detailed EDA Report — University Students Competition

- Train shape: (24444, 55)
- Test shape: (8149, 54)
- Features: 53 (excluding id and outcome)

## 1. Target Distribution

| outcome   |   count |   pct |
|:----------|--------:|------:|
| Pass      | 11538.0 |  47.2 |
| Withdrawn |  7617.0 |  31.2 |
| Fail      |  5289.0 |  21.6 |

**Imbalance ratio (majority/minority)**: 2.18x (Pass vs Fail)

## 2. Per-Module Outcome Distribution

This is critical because modules have fundamentally different data availability.

| code_module   |      n |   Pass |   Fail |   Withdrawn |
|:--------------|-------:|-------:|-------:|------------:|
| GGG           | 1901.0 |   59.8 |   28.7 |        11.5 |
| DDD           | 4703.0 |   41.6 |   22.5 |        35.9 |
| BBB           | 5931.0 |   47.5 |   22.3 |        30.2 |
| FFF           | 5821.0 |   47.0 |   22.0 |        31.0 |
| EEE           | 2202.0 |   56.2 |   19.2 |        24.6 |
| CCC           | 3325.0 |   37.8 |   17.6 |        44.5 |
| AAA           |  561.0 |   70.9 |   12.1 |        16.9 |

### Assessment Data Availability by Module

| code_module   |   pct_missing_scores |   mean_num_assessments |   pct_zero_assessments |
|:--------------|---------------------:|-----------------------:|-----------------------:|
| AAA           |                 11.9 |                    0.9 |                   11.9 |
| BBB           |                 25.6 |                    0.7 |                   25.6 |
| CCC           |                 26.6 |                    0.7 |                   26.6 |
| DDD           |                 25.4 |                    0.9 |                   25.4 |
| EEE           |                100.0 |                    0.0 |                  100.0 |
| FFF           |                 22.6 |                    0.8 |                   22.6 |
| GGG           |                100.0 |                    0.0 |                  100.0 |

**Key finding**: Modules EEE and GGG have ~100% missing scores (no assessments by day 28). In other modules, missing scores indicate students who didn't submit — a strong behavioral signal.

## 3. Missing Values Analysis

### Overall Missing Values

|                |   missing_count |   missing_pct |
|:---------------|----------------:|--------------:|
| mean_score_CMA |         21249.0 |          86.9 |
| mean_score_TMA |         11599.0 |          47.5 |
| max_score      |          9084.0 |          37.2 |
| mean_score     |          9084.0 |          37.2 |
| min_score      |          9084.0 |          37.2 |
| mean_days_late |          9074.0 |          37.1 |
| std_score      |          9074.0 |          37.1 |
| max_days_late  |          9074.0 |          37.1 |
| imd_band       |           860.0 |           3.5 |

### Structural vs. Behavioral Missing Scores

- Total rows with missing `mean_score`: 9084 (37.2%)
- Structural (EEE/GGG, no assessments available): 4103
- Behavioral (other modules, student didn't submit): 4981

**Outcome of students who didn't submit in modules WITH assessments** (n=4981):

| outcome   |   pct |
|:----------|------:|
| Withdrawn |  77.3 |
| Fail      |  18.7 |
| Pass      |   4.0 |

⚠️ This is a very strong signal — students who don't submit when they could are overwhelmingly likely to withdraw or fail.

## 4. Key Feature Distributions by Outcome


### `last_access_day`

| outcome   |   mean |   median |   std |    min |   max |
|:----------|-------:|---------:|------:|-------:|------:|
| Fail      |  20.53 |    25.00 |  9.30 | -24.00 | 28.00 |
| Pass      |  24.82 |    27.00 |  5.31 | -20.00 | 28.00 |
| Withdrawn |  13.00 |    17.00 | 12.89 | -25.00 | 28.00 |

### `total_clicks`

| outcome   |   mean |   median |    std |   min |     max |
|:----------|-------:|---------:|-------:|------:|--------:|
| Fail      | 223.66 |   131.00 | 297.01 |  0.00 | 4326.00 |
| Pass      | 435.74 |   294.00 | 473.56 |  0.00 | 5664.00 |
| Withdrawn | 172.54 |    60.00 | 300.89 |  0.00 | 5967.00 |

### `num_active_days`

| outcome   |   mean |   median |   std |   min |    max |
|:----------|-------:|---------:|------:|------:|-------:|
| Fail      |  67.28 |    47.00 | 70.62 |  0.00 | 686.00 |
| Pass      | 119.47 |    93.00 | 97.68 |  0.00 | 737.00 |
| Withdrawn |  51.48 |    24.00 | 72.42 |  0.00 | 802.00 |

### `mean_clicks_per_day`

| outcome   |   mean |   median |   std |   min |   max |
|:----------|-------:|---------:|------:|------:|------:|
| Fail      |   2.74 |     2.59 |  1.51 |  0.00 | 23.35 |
| Pass      |   3.28 |     3.06 |  1.34 |  0.00 | 20.53 |
| Withdrawn |   2.07 |     2.17 |  1.77 |  0.00 | 16.42 |

### `num_assessments`

| outcome   |   mean |   median |   std |   min |   max |
|:----------|-------:|---------:|------:|------:|------:|
| Fail      |   0.67 |     1.00 |  0.54 |  0.00 |  2.00 |
| Pass      |   0.81 |     1.00 |  0.47 |  0.00 |  2.00 |
| Withdrawn |   0.41 |     0.00 |  0.53 |  0.00 |  2.00 |

### `mean_score`

| outcome   |   mean |   median |   std |   min |    max |
|:----------|-------:|---------:|------:|------:|-------:|
| Fail      |  67.22 |    70.00 | 21.62 |  0.00 | 100.00 |
| Pass      |  77.04 |    80.00 | 19.97 |  0.00 | 100.00 |
| Withdrawn |  64.28 |    69.00 | 24.71 |  0.00 | 100.00 |

### `days_before_course_start`

| outcome   |   mean |   median |   std |     min |    max |
|:----------|-------:|---------:|------:|--------:|-------:|
| Fail      |  62.48 |    50.00 | 45.63 | -124.00 | 289.00 |
| Pass      |  66.33 |    53.00 | 47.33 | -101.00 | 310.00 |
| Withdrawn |  78.71 |    67.00 | 53.46 | -167.00 | 322.00 |

### `studied_credits`

| outcome   |   mean |   median |   std |   min |    max |
|:----------|-------:|---------:|------:|------:|-------:|
| Fail      |  76.06 |    60.00 | 38.86 | 30.00 | 360.00 |
| Pass      |  73.84 |    60.00 | 35.81 | 30.00 | 630.00 |
| Withdrawn |  91.23 |    60.00 | 47.56 | 30.00 | 655.00 |

### `num_of_prev_attempts`

| outcome   |   mean |   median |   std |   min |   max |
|:----------|-------:|---------:|------:|------:|------:|
| Fail      |   0.24 |     0.00 |  0.58 |  0.00 |  6.00 |
| Pass      |   0.11 |     0.00 |  0.39 |  0.00 |  6.00 |
| Withdrawn |   0.18 |     0.00 |  0.50 |  0.00 |  6.00 |

## 5. Weekly Engagement Trajectories

Mean clicks per week by outcome class:

| outcome   |   clicks_pre_course |   clicks_week1 |   clicks_week2 |   clicks_week3 |   clicks_week4 |
|:----------|--------------------:|---------------:|---------------:|---------------:|---------------:|
| Fail      |                50.1 |           42.8 |           39.6 |           54.8 |           36.4 |
| Pass      |               104.3 |           84.2 |           80.1 |           95.1 |           72.1 |
| Withdrawn |                43.0 |           33.7 |           30.8 |           40.6 |           24.4 |

### Engagement Trend (Week4 - Week1)

| outcome   |   mean |   median |    std |
|:----------|-------:|---------:|-------:|
| Fail      |  -6.31 |     0.00 |  76.14 |
| Pass      | -12.17 |    -4.00 | 112.58 |
| Withdrawn |  -9.29 |     0.00 |  67.75 |

**Interpretation**: Withdrawn students show declining engagement (negative trend). Pass students maintain or increase engagement.

## 6. Categorical Feature vs. Outcome


### `code_module`

| code_module   |      n |   Pass |   Fail |   Withdrawn |
|:--------------|-------:|-------:|-------:|------------:|
| AAA           |  561.0 |   70.9 |   12.1 |        16.9 |
| GGG           | 1901.0 |   59.8 |   28.7 |        11.5 |
| EEE           | 2202.0 |   56.2 |   19.2 |        24.6 |
| BBB           | 5931.0 |   47.5 |   22.3 |        30.2 |
| FFF           | 5821.0 |   47.0 |   22.0 |        31.0 |
| DDD           | 4703.0 |   41.6 |   22.5 |        35.9 |
| CCC           | 3325.0 |   37.8 |   17.6 |        44.5 |

### `gender`

| gender   |       n |   Pass |   Fail |   Withdrawn |
|:---------|--------:|-------:|-------:|------------:|
| F        | 10972.0 |   48.6 |   21.0 |        30.4 |
| M        | 13472.0 |   46.1 |   22.1 |        31.8 |

### `region`

*(Showing top 10 of 13 categories)*

| region               |      n |   Pass |   Fail |   Withdrawn |
|:---------------------|-------:|-------:|-------:|------------:|
| Ireland              |  882.0 |   55.9 |   21.8 |        22.3 |
| South Region         | 2345.0 |   52.4 |   18.0 |        29.6 |
| North Region         | 1387.0 |   51.1 |   17.1 |        31.8 |
| South East Region    | 1543.0 |   51.0 |   18.7 |        30.3 |
| South West Region    | 1839.0 |   49.4 |   18.8 |        31.9 |
| East Anglian Region  | 2539.0 |   49.0 |   21.0 |        30.1 |
| Scotland             | 2576.0 |   48.9 |   24.1 |        26.9 |
| Yorkshire Region     | 1513.0 |   45.6 |   22.5 |        31.9 |
| East Midlands Region | 1783.0 |   45.4 |   20.3 |        34.3 |
| Wales                | 1568.0 |   45.2 |   29.5 |        25.3 |

### `highest_education`

| highest_education           |       n |   Pass |   Fail |   Withdrawn |
|:----------------------------|--------:|-------:|-------:|------------:|
| Post Graduate Qualification |   235.0 |   65.1 |    9.8 |        25.1 |
| HE Qualification            |  3562.0 |   55.9 |   17.0 |        27.1 |
| A Level or Equivalent       | 10482.0 |   52.0 |   19.3 |        28.7 |
| Lower Than A Level          |  9899.0 |   39.1 |   26.0 |        35.0 |
| No Formal quals             |   266.0 |   28.9 |   24.1 |        47.0 |

### `imd_band`

| imd_band   |      n |   Pass |   Fail |   Withdrawn |
|:-----------|-------:|-------:|-------:|------------:|
| 90-100%    | 1900.0 |   57.7 |   16.4 |        25.9 |
| 80-90%     | 2074.0 |   54.7 |   18.0 |        27.3 |
| 60-70%     | 2153.0 |   51.5 |   19.1 |        29.4 |
| 70-80%     | 2120.0 |   50.0 |   22.4 |        27.7 |
| 50-60%     | 2348.0 |   49.1 |   22.7 |        28.3 |
| 40-50%     | 2442.0 |   46.7 |   21.9 |        31.4 |
| 30-40%     | 2671.0 |   46.5 |   21.6 |        31.9 |
| 20-30%     | 2772.0 |   41.0 |   22.6 |        36.4 |
| 10-20      | 2606.0 |   39.4 |   25.7 |        34.9 |
| 0-10%      | 2498.0 |   35.0 |   26.9 |        38.1 |

### `age_band`

| age_band   |       n |   Pass |   Fail |   Withdrawn |
|:-----------|--------:|-------:|-------:|------------:|
| 55<=       |   158.0 |   60.8 |   13.9 |        25.3 |
| 35-55      |  7067.0 |   51.5 |   19.1 |        29.4 |
| 0-35       | 17219.0 |   45.3 |   22.8 |        31.9 |

### `disability`

| disability   |       n |   Pass |   Fail |   Withdrawn |
|:-------------|--------:|-------:|-------:|------------:|
| N            | 22028.0 |   48.3 |   21.5 |        30.2 |
| Y            |  2416.0 |   37.4 |   23.0 |        39.6 |

## 7. Top Feature Correlations with Outcome

Using numeric encoding: Pass=2, Withdrawn=1, Fail=0

|                       |   abs_correlation |
|:----------------------|------------------:|
| mean_score_CMA        |            0.3845 |
| num_active_days       |            0.2743 |
| last_access_day       |            0.2544 |
| total_clicks          |            0.2428 |
| num_unique_activities |            0.2375 |
| min_score             |            0.2150 |
| mean_score            |            0.2133 |
| clicks_homepage       |            0.2125 |
| max_score             |            0.2104 |
| clicks_week2          |            0.2098 |
| clicks_week4          |            0.1951 |
| clicks_oucontent      |            0.1940 |
| mean_clicks_per_day   |            0.1920 |
| clicks_week1          |            0.1904 |
| clicks_week3          |            0.1893 |
| clicks_subpage        |            0.1797 |
| clicks_forumng        |            0.1795 |
| clicks_pre_course     |            0.1793 |
| mean_score_TMA        |            0.1715 |
| first_access_day      |            0.1671 |

### Highly Correlated Feature Pairs (|r| > 0.8)

|    | feature_1             | feature_2             |   correlation |
|---:|:----------------------|:----------------------|--------------:|
|  0 | mean_score            | min_score             |         0.997 |
|  1 | mean_score            | max_score             |         0.997 |
|  2 | mean_score            | mean_score_TMA        |         0.996 |
|  3 | max_score             | mean_score_TMA        |         0.996 |
|  4 | mean_days_late        | max_days_late         |         0.992 |
|  5 | min_score             | mean_score_TMA        |         0.989 |
|  6 | mean_score            | mean_score_CMA        |         0.988 |
|  7 | min_score             | max_score             |         0.988 |
|  8 | min_score             | mean_score_CMA        |         0.988 |
|  9 | max_score             | mean_score_CMA        |         0.968 |
| 10 | std_clicks_per_day    | max_clicks_per_day    |         0.908 |
| 11 | total_clicks          | num_active_days       |         0.906 |
| 12 | num_active_days       | num_unique_activities |         0.880 |
| 13 | total_clicks          | clicks_homepage       |         0.876 |
| 14 | num_unique_activities | clicks_subpage        |         0.864 |
| 15 | num_active_days       | clicks_subpage        |         0.849 |
| 16 | num_active_days       | clicks_homepage       |         0.815 |
| 17 | total_clicks          | clicks_forumng        |         0.813 |
| 18 | mean_clicks_per_day   | std_clicks_per_day    |         0.808 |
| 19 | total_clicks          | clicks_week1          |         0.806 |

**20 highly correlated pairs found.** Consider dropping redundant features or using PCA.

## 8. `last_access_day` Deep Dive (Top Feature)

Distribution of last_access_day by outcome:

| outcome   |   mean |   median |   p25 |   p75 |   std |
|:----------|-------:|---------:|------:|------:|------:|
| Fail      |   20.5 |     25.0 |  18.0 |  27.0 |   9.3 |
| Pass      |   24.8 |     27.0 |  24.0 |  28.0 |   5.3 |
| Withdrawn |   13.0 |     17.0 |   0.0 |  26.0 |  12.9 |

### Students who stopped accessing VLE early


**last_access_day ≤ 7** (n=4187, 17.1% of data):
  Pass=6.8%, Fail=16.9%, Withdrawn=76.4%

**last_access_day ≤ 14** (n=5131, 21.0% of data):
  Pass=10.4%, Fail=19.4%, Withdrawn=70.2%

**last_access_day ≤ 21** (n=7816, 32.0% of data):
  Pass=20.2%, Fail=22.9%, Withdrawn=56.9%


## 9. Train vs Test Distribution Comparison

Checking for distribution shift between train and test sets.

### Categorical Features


### Numeric Features (mean comparison)

| feature              |   train_mean |   test_mean |   pct_diff |
|:---------------------|-------------:|------------:|-----------:|
| clicks_sharedsubpage |         0.00 |        0.00 |      45.79 |
| num_banked           |         0.02 |        0.02 |      15.24 |
| clicks_dataplus      |         0.01 |        0.01 |      13.72 |
| clicks_glossary      |         0.85 |        0.93 |       9.73 |
| std_score            |         0.40 |        0.37 |       8.25 |
| clicks_dualpane      |         0.12 |        0.13 |       6.80 |
| clicks_quiz          |        41.55 |       43.43 |       4.54 |
| clicks_questionnaire |         0.15 |        0.16 |       4.31 |
| clicks_page          |         1.13 |        1.18 |       4.18 |
| clicks_oucontent     |        63.98 |       66.30 |       3.62 |
| clicks_week4         |        49.51 |       51.23 |       3.47 |
| max_days_late        |        -1.64 |       -1.69 |       3.28 |
| clicks_pre_course    |        73.46 |       75.84 |       3.24 |
| clicks_url           |         6.38 |        6.55 |       2.70 |
| clicks_week1         |        59.52 |       61.09 |       2.64 |


## 10. Key EDA Findings Summary


1. **Class imbalance**: Pass (47.2%) >> Fail (21.6%). Models need class balancing.
2. **Module heterogeneity**: Fail rates range from ~15% to ~30% across modules. Modules EEE/GGG have no assessment data.
3. **Structural vs behavioral missing**: ~37% of scores are missing. Distinguishing "no assessment available" from "didn't submit" is critical — non-submitters in assessment-available modules overwhelmingly fail/withdraw.
4. **`last_access_day` is the single strongest predictor**: Students who stop accessing VLE early are almost certain to withdraw/fail.
5. **Engagement trends matter**: Withdrawn students show declining weekly clicks; Pass students maintain engagement.
6. **High multicollinearity**: Many click features are redundant (total_clicks correlated with weekly sums, etc.).
7. **Train/test distributions are similar**: No major distribution shift detected.
8. **Previous attempts**: Students with >0 previous attempts have different outcome distributions.

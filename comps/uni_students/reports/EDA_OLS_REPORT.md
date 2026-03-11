# EDA and Simple OLS for University Students Competition

## Data Loading
- Loaded train data: (24444, 55)
- Loaded test data: (8149, 54)

## Exploratory Data Analysis (EDA)

### Target Variable Distribution (`outcome`)
| outcome   |   proportion |
|:----------|-------------:|
| Pass      |     0.472018 |
| Withdrawn |     0.31161  |
| Fail      |     0.216372 |

Observations:
- The target variable is categorical with 3 classes: Pass, Withdrawn, Fail.
- 'Pass' is likely the majority class or significant portion.

### Missing Values
|                |   Count |
|:---------------|--------:|
| mean_score_CMA |   21249 |
| mean_score_TMA |   11599 |
| max_score      |    9084 |
| mean_score     |    9084 |
| min_score      |    9084 |
| mean_days_late |    9074 |
| std_score      |    9074 |
| max_days_late  |    9074 |
| imd_band       |     860 |

Observations:
- Several columns have missing values, particularly assessment scores.
- This is expected as per data description (some courses have no assessments in first 4 weeks).

### Numeric Features Summary
|                          |   count |         mean |         std |   min |        25% |         50% |         75% |        max |
|:-------------------------|--------:|-------------:|------------:|------:|-----------:|------------:|------------:|-----------:|
| id                       |   24444 | 12221.5      | 7056.52     |     0 | 6110.75    | 12221.5     | 18332.2     | 24443      |
| num_of_prev_attempts     |   24444 |     0.162658 |    0.476218 |     0 |    0       |     0       |     0       |     6      |
| studied_credits          |   24444 |    79.7365   |   41.1981   |    30 |   60       |    60       |   120       |   655      |
| days_before_course_start |   24444 |    69.3551   |   49.3994   |  -167 |   29       |    57       |   100       |   322      |
| total_clicks             |   24444 |   307.837    |  410.005    |     0 |   48       |   178       |   412       |  5967      |
| mean_clicks_per_day      |   24444 |     2.78687  |    1.61121  |     0 |    2       |     2.70732 |     3.65403 |    23.3478 |
| std_clicks_per_day       |   24444 |     4.18875  |    4.55907  |     0 |    1.68161 |     3.07859 |     5.79208 |   253.104  |
| max_clicks_per_day       |   24444 |    29.5658   |   46.8984   |     0 |    7       |    17       |    42       |  4098      |
| first_access_day         |   24444 |    -8.18459  |    9.22609  |   -25 |  -17       |    -9       |     0       |    28      |
| last_access_day          |   24444 |    20.2071   |   10.4904   |   -25 |   18       |    25       |    27       |    28      |

Observations:
- Features have different scales (e.g., clicks vs scores).
- Some features like `num_of_prev_attempts` are low-range integers.

## Simple OLS Model

Since the target is categorical (3 classes), we will perform a **Linear Probability Model (LPM)** by converting the target to binary: **Pass (1) vs. Fail/Withdrawn (0)**.

### Model Training
- Features used: 83
- Target: Binary (Pass=1, Other=0)

### OLS Regression Results (Summary)
- **R-squared**: 0.3369
- **Adj. R-squared**: 0.3348
- **F-statistic**: 156.6923
- **Prob (F-statistic)**: 0.0000e+00

### Significant Features (p < 0.05)
|                                      |      P-Value |
|:-------------------------------------|-------------:|
| num_assessments                      | 6.13634e-165 |
| last_access_day                      | 4.45092e-64  |
| highest_education_Lower Than A Level | 5.09026e-61  |
| code_module_GGG                      | 2.08001e-43  |
| num_active_days                      | 3.06199e-25  |
| code_module_DDD                      | 7.87602e-22  |
| num_banked                           | 4.41649e-20  |
| const                                | 5.00155e-18  |
| code_module_FFF                      | 2.88714e-16  |
| imd_band_90-100%                     | 7.54364e-16  |

### Observations from OLS:
- R-squared indicates how much variance in the binary outcome is explained by the features.
- Significant features suggest which variables are most strongly associated with passing.
- Note: OLS predictions can fall outside [0, 1], which is a limitation of LPM.

## Submission
- Generated submission file: `data/uni_students/submission_ols.csv`
- Strategy: OLS > 0.5 -> 'Pass', else 'Fail'. (Binary simplification)

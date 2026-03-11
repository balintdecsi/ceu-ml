# Extended EDA Summary

## Scope
- Focused on module heterogeneity, missingness, engagement trajectories, assessment behavior, and feature redundancy.
- Chosen to directly address the bottlenecks documented in `reports/ANALYSIS_AND_IDEAS.md`: low Fail recall, structural missingness, and over-reliance on a single validation split.

## Data Overview
- Training rows: 24,444
- Columns: 55

### Outcome Distribution
| outcome   |   proportion |
|:----------|-------------:|
| Pass      |        0.472 |
| Withdrawn |        0.312 |
| Fail      |        0.216 |

### Outcome Mix by Module
| code_module   |   Fail |   Pass |   Withdrawn |
|:--------------|-------:|-------:|------------:|
| AAA           |  0.121 |  0.709 |       0.169 |
| BBB           |  0.223 |  0.475 |       0.302 |
| CCC           |  0.176 |  0.378 |       0.445 |
| DDD           |  0.225 |  0.416 |       0.359 |
| EEE           |  0.192 |  0.562 |       0.246 |
| FFF           |  0.22  |  0.47  |       0.31  |
| GGG           |  0.287 |  0.598 |       0.115 |

### Assessment Missingness by Module
| code_module   |   mean_score |   mean_score_TMA |   mean_score_CMA |
|:--------------|-------------:|-----------------:|-----------------:|
| AAA           |        0.119 |            0.119 |            1     |
| BBB           |        0.256 |            0.256 |            1     |
| CCC           |        0.266 |            1     |            0.266 |
| DDD           |        0.254 |            0.27  |            0.84  |
| EEE           |        1     |            1     |            1     |
| FFF           |        0.226 |            0.226 |            1     |
| GGG           |        1     |            1     |            1     |

### Mean Weekly Click Trajectory by Outcome
| outcome   |   clicks_week1 |   clicks_week2 |   clicks_week3 |   clicks_week4 |
|:----------|---------------:|---------------:|---------------:|---------------:|
| Fail      |           42.8 |           39.6 |           54.8 |           36.4 |
| Pass      |           84.2 |           80.1 |           95.1 |           72.1 |
| Withdrawn |           33.7 |           30.8 |           40.6 |           24.4 |

### `last_access_day` Quartiles by Outcome
| outcome   |   0.25 |   0.5 |   0.75 |
|:----------|-------:|------:|-------:|
| Fail      |     18 |    25 |     27 |
| Pass      |     24 |    27 |     28 |
| Withdrawn |      0 |    17 |     26 |

### Mean Assessment / Submission Behavior by Outcome
| outcome   |   mean_score |   max_score |   mean_days_late |   num_assessments |
|:----------|-------------:|------------:|-----------------:|------------------:|
| Fail      |        67.22 |       67.62 |            -1.7  |              0.67 |
| Pass      |        77.04 |       77.27 |            -1.98 |              0.81 |
| Withdrawn |        64.28 |       64.58 |            -1.06 |              0.41 |

### Strongest Numeric Correlations
| feature_a      | feature_b      |   abs_correlation |
|:---------------|:---------------|------------------:|
| mean_score     | min_score      |          0.997092 |
| mean_score     | max_score      |          0.997049 |
| mean_score     | mean_score_TMA |          0.996076 |
| max_score      | mean_score_TMA |          0.995785 |
| mean_days_late | max_days_late  |          0.992178 |
| min_score      | mean_score_TMA |          0.988874 |
| mean_score     | mean_score_CMA |          0.988329 |
| min_score      | max_score      |          0.988299 |
| min_score      | mean_score_CMA |          0.987805 |
| max_score      | mean_score_CMA |          0.967919 |

## Modeling Implications
- Keep module identity explicit because both outcomes and assessment availability vary materially by module.
- Preserve missingness as signal: all-score-missing rows need to distinguish structural no-assessment modules from behavioural non-submission.
- Compare models with stratified cross-validation instead of a single holdout because the Fail class is the scarcest class.
- Retain a neural-network baseline only after feature scaling, while tree models should use the unscaled engineered feature matrix.

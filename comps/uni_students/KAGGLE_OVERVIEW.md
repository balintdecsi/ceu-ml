Overview
The Scenario

It's week 4 of the semester. You're part of the university's student support team. Your job: identify which students need intervention right now before it's too late.

You have data on students from the Open University Learning Analytics Dataset (OULAD), but only what's available through day 28 (end of week 4): demographics, early VLE engagement, and the first round of assessment scores.

Can you predict who will Pass, Fail, or Withdraw using only early signal data?
Prediction Task

Predict the outcome for each student in the test set.

Target Variable: outcome (3 classes)

    Pass (47.2%): Student passed the course (includes distinction)
    Withdrawn (31.2%): Student withdrew before completion
    Fail (21.6%): Student failed the course

Features at a Glance

The dataset includes 53 features across three categories (see the Data tab for full documentation of every feature):

    Demographic & Registration (10 features): Course code, gender, region, education level, socioeconomic indicator, age, previous attempts, credits, disability, registration timing

    VLE Engagement (31 features): Click data through day 28 — total clicks, weekly patterns (weeks 1-4), clicks by activity type (forums, content, quizzes, etc.)

    Assessment Performance (12 features): Only assessments due and submitted by day 28 — scores on Tutor Marked Assessments (TMAs) and Computer Marked Assessments (CMAs), submission timing, late/early counts.

Tips

    Start with a simple baseline (always predicting "Pass" gives 47.2% accuracy).
    Pay attention to missing values as they carry information.
    Not all modules have the same data available by week 4.
    The Fail class is the hardest to predict so don't be surprised by low recall.

Citation

Kuzilek J., Hlosta M., Zdrahal Z. Open University Learning Analytics dataset, Sci.Data 4:170171 (2017).

Evaluation
Evaluation Metric

$\text{Accuracy} = \frac{\text{Number of correct predictions}}{\text{Total predictions}}$

Submissions are evaluated on categorization accuracy: the proportion of test observations where the predicted class exactly matches the true class.

Submission Format

Submit a CSV file with exactly 2 columns:

ID,outcome
0,Pass
1,Withdrawn
2,Fail

    The id column must match the IDs in test.csv
    The outcome column must be one of: Pass, Withdrawn, or Fail
    The file must contain a header row
    There must be exactly 8,149 predictions (one per test observation)

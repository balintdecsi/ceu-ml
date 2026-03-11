Dataset Description
About the Data

This dataset is derived from the Open University Learning Analytics Dataset (OULAD): real anonymized data from ~30,000 students across 7 courses at the UK's Open University.

Original Source: Kuzilek J., Hlosta M., Zdrahal Z. Open University Learning Analytics dataset, Sci. Data 4:170171 doi: 10.1038/sdata.2017.171 (2017).
What is a row?

Each row represents one student-in-one-course. The same physical student taking multiple courses (or retaking the same course in a different semester) appears as separate rows. For simplicity, we will handle them as separate observations.
Temporal cutoff

All behavioral data (VLE clicks, assessment scores) is restricted to day ≤ 28 so the first 4 weeks of the course. The scenario is: it's end of week 4, and you only know what has happened so far.
Files

    train.csv: Training data with 24,444 rows and 55 columns (53 features + ID + outcome)
    test.csv: Test data with 8,149 rows and 54 columns (53 features + ID, no outcome)
    sample_submission.csv: Example submission file predicting "Pass" for all rows (47.2% accuracy baseline)

Features (53 total)
1. Demographic & Registration (10 features)
Feature 	Type 	Description
code_module 	Categorical 	Course code (AAA, BBB, CCC, DDD, EEE, FFF, GGG). Different courses have very different structures — see "Module Variation" below.
gender 	Categorical 	M / F
region 	Categorical 	UK geographic region (13 values)
highest_education 	Categorical 	Prior education level (e.g., "A Level or Equivalent", "HE Qualification", "Lower Than A Level", "No Formal quals", "Post Graduate Qualification")
imd_band 	Categorical 	Index of Multiple Deprivation — a UK socioeconomic indicator. "0-10%" = most deprived area, "90-100%" = least deprived.
age_band 	Categorical 	Age group: 0-35, 35-55, 55<=
num_of_prev_attempts 	Numeric 	Number of previous attempts at this same course (0–6). 0 = first attempt.
studied_credits 	Numeric 	Total credits the student is studying across all courses this semester
disability 	Categorical 	Y / N — whether the student declared a disability
days_before_course_start 	Numeric 	How many days before the course started the student registered. Positive = registered early (most students). Negative = registered after the course already began (rare).
2. VLE Engagement (31 features)

These measure how students interact with the Virtual Learning Environment (the university's online platform similar to Moodle) during the first 4 weeks.
Overall engagement (8 features):
Feature 	Description
total_clicks 	Total clicks across all activities through day 28
mean_clicks_per_day 	Average clicks per active day
std_clicks_per_day 	Standard deviation of daily clicks
max_clicks_per_day 	Highest number of clicks in a single day
num_active_days 	Number of distinct days the student accessed the VLE
num_unique_activities 	Number of different learning activities accessed
first_access_day 	Day of first VLE access (can be negative = before course start)
last_access_day 	Day of last VLE access (max = 28)
Weekly click patterns (5 features):
Feature 	Description
clicks_pre_course 	Clicks before the course officially started (day < 0)
clicks_week1 	Clicks during days 1–7
clicks_week2 	Clicks during days 8–14
clicks_week3 	Clicks during days 15–21
clicks_week4 	Clicks during days 22–28
Activity type clicks (18 features):

Each feature counts clicks on a specific type of learning activity:
Feature 	What it is
Content delivery 	
clicks_oucontent 	Structured reading material — the main course content (highest click volume for most students)
clicks_resource 	Downloadable files (PDFs, documents)
clicks_subpage 	Sub-pages linking to other content
clicks_page 	Simple HTML content pages
clicks_homepage 	Course landing page
clicks_url 	External web links
clicks_dualpane 	Side-by-side content viewer
clicks_sharedsubpage 	Sub-pages shared across modules (very sparse)
Assessment & practice 	
clicks_quiz 	Online quizzes and self-assessments
clicks_externalquiz 	Quizzes hosted outside the VLE
clicks_questionnaire 	Surveys and feedback forms (not graded)
Collaboration 	
clicks_forumng 	Discussion forums
clicks_oucollaborate 	Real-time virtual classroom sessions
clicks_ouelluminate 	Live online tutorials (Blackboard Collaborate)
clicks_ouwiki 	Collaborative wiki pages
Other 	
clicks_dataplus 	Structured data entry tool
clicks_glossary 	Key terms dictionary
clicks_htmlactivity 	Custom HTML interactive content
3. Assessment Performance (12 features)

Only assessments that were both due by day 28 AND submitted by day 28 are included. Most students have 0–2 assessments in this window. Some modules have no assessments due by day 28 at all (see "Module Variation" below).

The Open University uses two assessment types:

    TMA (Tutor Marked Assessment): Essays or assignments graded by a human tutor
    CMA (Computer Marked Assessment): Automatically graded quizzes/tests

Feature 	Description
mean_score 	Average score across all submitted assessments (0–100 scale)
std_score 	Standard deviation of scores (0 if only one assessment)
min_score 	Lowest score received
max_score 	Highest score received
num_assessments 	Number of assessments submitted by day 28 (0 = no submissions)
num_banked 	Number of assessments carried over ("banked") from a previous attempt at the course
mean_score_TMA 	Average score on Tutor Marked Assessments only
mean_score_CMA 	Average score on Computer Marked Assessments only
mean_days_late 	Average days late across submissions. Negative = submitted early on average.
max_days_late 	Latest submission relative to deadline. Negative = all were early.
num_late_submissions 	Count of submissions made after the deadline
num_early_submissions 	Count of submissions made more than 7 days before the deadline
Important Data Notes
Missing Values

~37% of rows have no assessment data. Their score columns (mean_score, std_score, min_score, max_score, mean_score_TMA, mean_score_CMA, mean_days_late, max_days_late) are missing. This happens for two reasons:

    Some modules have no assessments due by day 28. Modules EEE and GGG simply don't have any assignments scheduled in the first 4 weeks.
    Some students didn't submit. In other modules (AAA, BBB, CCC, DDD, FFF), there ARE assessments due by day 28, but some students haven't turned anything in.

Think carefully about how you handle missing values. Most models can't work with them directly so you'll need to decide what to do: fill with a value, drop columns, create indicator flags, or use a model that handles missing data natively (like XGBoost).
Module Variation

Different courses have very different data availability by week 4:

    AAA, BBB, DDD have TMAs due before day 28 — most students have assessment scores
    EEE, GGG have NO assessments due by day 28 — all assessment score features are missing for these modules

Categorical Features

7 features are categorical and need encoding before use in most models: code_module, gender, region, highest_education, imd_band, age_band, disability

Common approaches: one-hot encoding, label encoding, target encoding, or using a model that handles categoricals natively (like CatBoost).
Files

3 files
Size

9.19 MB
Type

csv
License


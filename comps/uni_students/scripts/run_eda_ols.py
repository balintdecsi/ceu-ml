import pandas as pd
import numpy as np
import statsmodels.api as sm
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
import seaborn as sns

# Set pandas options for better display
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)

def print_markdown(text):
    print(text)

def main():
    print_markdown("# EDA and Simple OLS for University Students Competition\n")

    # Load Data
    train_path = 'data/uni_students/train.csv'
    test_path = 'data/uni_students/test.csv'
    
    try:
        train_df = pd.read_csv(train_path)
        test_df = pd.read_csv(test_path)
        print_markdown(f"## Data Loading\n- Loaded train data: {train_df.shape}\n- Loaded test data: {test_df.shape}\n")
    except FileNotFoundError:
        print_markdown("Error: Data files not found.")
        return

    # EDA
    print_markdown("## Exploratory Data Analysis (EDA)\n")
    
    print_markdown("### Target Variable Distribution (`outcome`)")
    outcome_counts = train_df['outcome'].value_counts(normalize=True)
    print_markdown(outcome_counts.to_markdown())
    print_markdown("\nObservations:\n- The target variable is categorical with 3 classes: Pass, Withdrawn, Fail.\n- 'Pass' is likely the majority class or significant portion.\n")

    print_markdown("### Missing Values")
    missing_vals = train_df.isnull().sum()
    missing_vals = missing_vals[missing_vals > 0].sort_values(ascending=False)
    if not missing_vals.empty:
        print_markdown(missing_vals.head(10).to_markdown(headers=["Count"]))
        print_markdown("\nObservations:\n- Several columns have missing values, particularly assessment scores.\n- This is expected as per data description (some courses have no assessments in first 4 weeks).\n")
    else:
        print_markdown("- No missing values found.\n")

    print_markdown("### Numeric Features Summary")
    numeric_cols = train_df.select_dtypes(include=[np.number]).columns
    print_markdown(train_df[numeric_cols].describe().T.head(10).to_markdown())
    print_markdown("\nObservations:\n- Features have different scales (e.g., clicks vs scores).\n- Some features like `num_of_prev_attempts` are low-range integers.\n")

    # Prepare for OLS
    print_markdown("## Simple OLS Model\n")
    print_markdown("Since the target is categorical (3 classes), we will perform a **Linear Probability Model (LPM)** by converting the target to binary: **Pass (1) vs. Fail/Withdrawn (0)**.\n")

    # 1. Target Encoding
    train_df['target_binary'] = (train_df['outcome'] == 'Pass').astype(int)
    
    # 2. Feature Selection & Preprocessing
    # Drop ID and original outcome
    X = train_df.drop(['id', 'outcome', 'target_binary'], axis=1)
    y = train_df['target_binary']
    
    # Identify categorical and numeric columns
    categorical_cols = X.select_dtypes(include=['object']).columns
    numeric_cols = X.select_dtypes(include=[np.number]).columns

    # Simple imputation
    # Numeric: mean
    # Categorical: constant 'missing' (or mode)
    
    # We'll use pandas get_dummies for simplicity in OLS to see feature names easily
    X_numeric = X[numeric_cols].fillna(X[numeric_cols].mean())
    X_categorical = X[categorical_cols].fillna('Missing')
    X_categorical_encoded = pd.get_dummies(X_categorical, drop_first=True) # drop_first to avoid multicollinearity
    
    # Boolean columns might be created, convert to int
    X_categorical_encoded = X_categorical_encoded.astype(int)

    X_final = pd.concat([X_numeric, X_categorical_encoded], axis=1)
    
    # Add constant for OLS
    X_final = sm.add_constant(X_final)
    
    print_markdown(f"### Model Training\n- Features used: {X_final.shape[1]}\n- Target: Binary (Pass=1, Other=0)\n")

    # 3. Run OLS
    model = sm.OLS(y, X_final)
    results = model.fit()
    
    print_markdown("### OLS Regression Results (Summary)")
    # Extract key metrics to print in markdown instead of full raw text which is messy
    print_markdown(f"- **R-squared**: {results.rsquared:.4f}")
    print_markdown(f"- **Adj. R-squared**: {results.rsquared_adj:.4f}")
    print_markdown(f"- **F-statistic**: {results.fvalue:.4f}")
    print_markdown(f"- **Prob (F-statistic)**: {results.f_pvalue:.4e}")
    
    print_markdown("\n### Significant Features (p < 0.05)")
    pvalues = results.pvalues
    significant = pvalues[pvalues < 0.05].sort_values()
    print_markdown(significant.head(10).to_frame(name="P-Value").to_markdown())
    
    print_markdown("\n### Observations from OLS:")
    print_markdown("- R-squared indicates how much variance in the binary outcome is explained by the features.")
    print_markdown("- Significant features suggest which variables are most strongly associated with passing.")
    print_markdown("- Note: OLS predictions can fall outside [0, 1], which is a limitation of LPM.")

    # Generate Predictions for Test Set
    # We need to replicate preprocessing for test set
    test_ids = test_df['id']
    X_test = test_df.drop(['id'], axis=1)
    
    X_test_numeric = X_test[numeric_cols].fillna(train_df[numeric_cols].mean()) # Use train mean!
    X_test_categorical = X_test[categorical_cols].fillna('Missing')
    X_test_categorical_encoded = pd.get_dummies(X_test_categorical, drop_first=True)
    
    # Align columns (ensure test has same columns as train, fill missing with 0)
    X_test_categorical_encoded = X_test_categorical_encoded.reindex(columns=X_categorical_encoded.columns, fill_value=0)
    X_test_categorical_encoded = X_test_categorical_encoded.astype(int)

    X_test_final = pd.concat([X_test_numeric, X_test_categorical_encoded], axis=1)
    X_test_final = sm.add_constant(X_test_final, has_constant='add')
    
    # Predict
    predictions_prob = results.predict(X_test_final)
    
    # Create submission (Simple rule: > 0.5 is Pass, else Fail)
    # Note: The competition has 3 classes. This is a binary approximation.
    predictions_class = np.where(predictions_prob > 0.5, 'Pass', 'Fail')
    
    submission_df = pd.DataFrame({'id': test_ids, 'outcome': predictions_class})
    submission_path = 'data/uni_students/submission_ols.csv'
    submission_df.to_csv(submission_path, index=False)
    
    print_markdown(f"\n## Submission\n- Generated submission file: `{submission_path}`")
    print_markdown("- Strategy: OLS > 0.5 -> 'Pass', else 'Fail'. (Binary simplification)")

if __name__ == "__main__":
    main()

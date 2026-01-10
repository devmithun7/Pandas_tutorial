"""
NULL IMPUTATION TECHNIQUES IN PANDAS
====================================

This file demonstrates various methods for handling missing values (NaN, None, null) 
in pandas DataFrames. Each technique is explained with comments and examples.

Author: Learning Pandas
Date: January 2026
"""

import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
import warnings
warnings.filterwarnings('ignore')

print("=" * 60)
print("NULL IMPUTATION TECHNIQUES DEMONSTRATION")
print("=" * 60)

# =============================================================================
# 1. CREATING SAMPLE DATA WITH MISSING VALUES
# =============================================================================

print("\n1. CREATING SAMPLE DATASET WITH MISSING VALUES")
print("-" * 50)

# Create a sample dataset with intentional missing values
np.random.seed(42)  # For reproducible results

data = {
    'Name': ['Alice', 'Bob', 'Charlie', 'Diana', 'Eve', 'Frank', 'Grace', 'Henry'],
    'Age': [25, np.nan, 35, 28, np.nan, 45, 32, np.nan],
    'Salary': [50000, 60000, np.nan, 55000, np.nan, 80000, 65000, 70000],
    'Department': ['IT', 'HR', np.nan, 'IT', 'Finance', np.nan, 'HR', 'IT'],
    'Experience': [2, 5, np.nan, 3, 7, np.nan, 4, 8],
    'Rating': [4.2, np.nan, 3.8, 4.5, np.nan, 4.0, 4.3, np.nan]
}

df_original = pd.DataFrame(data)
print("Original dataset with missing values:")
print(df_original)
print(f"\nMissing values per column:")
print(df_original.isnull().sum())

# =============================================================================
# 2. BASIC IMPUTATION METHODS
# =============================================================================

print("\n\n2. BASIC IMPUTATION METHODS")
print("-" * 50)

# 2.1 DROP MISSING VALUES
print("\n2.1 DROPPING MISSING VALUES")
print("Method: Remove rows/columns with missing values")

df_drop_rows = df_original.dropna()  # Drop rows with any missing value
print(f"After dropping rows with missing values: {df_drop_rows.shape[0]} rows remaining")
print(df_drop_rows)

df_drop_cols = df_original.dropna(axis=1)  # Drop columns with any missing value
print(f"\nAfter dropping columns with missing values: {df_drop_cols.shape[1]} columns remaining")
print(df_drop_cols.columns.tolist())

# Drop rows only if ALL values are missing
df_drop_all = df_original.dropna(how='all')
print(f"\nAfter dropping rows where ALL values are missing: {df_drop_all.shape[0]} rows")

# 2.2 FILL WITH CONSTANT VALUES
print("\n\n2.2 FILLING WITH CONSTANT VALUES")
print("Method: Replace missing values with a specific constant")

df_fill_zero = df_original.copy()
df_fill_zero['Age'] = df_fill_zero['Age'].fillna(0)
df_fill_zero['Salary'] = df_fill_zero['Salary'].fillna(0)
print("Filling numerical columns with 0:")
print(df_fill_zero[['Name', 'Age', 'Salary']])

df_fill_unknown = df_original.copy()
df_fill_unknown['Department'] = df_fill_unknown['Department'].fillna('Unknown')
print("\nFilling categorical columns with 'Unknown':")
print(df_fill_unknown[['Name', 'Department']])

# 2.3 MEAN IMPUTATION
print("\n\n2.3 MEAN IMPUTATION")
print("Method: Replace missing values with the mean of the column")
print("Best for: Numerical data with normal distribution")

df_mean = df_original.copy()
# Calculate mean for numerical columns
age_mean = df_mean['Age'].mean()
salary_mean = df_mean['Salary'].mean()
experience_mean = df_mean['Experience'].mean()
rating_mean = df_mean['Rating'].mean()

print(f"Age mean: {age_mean:.2f}")
print(f"Salary mean: {salary_mean:.2f}")
print(f"Experience mean: {experience_mean:.2f}")
print(f"Rating mean: {rating_mean:.2f}")

# Fill missing values with mean
df_mean['Age'] = df_mean['Age'].fillna(age_mean)
df_mean['Salary'] = df_mean['Salary'].fillna(salary_mean)
df_mean['Experience'] = df_mean['Experience'].fillna(experience_mean)
df_mean['Rating'] = df_mean['Rating'].fillna(rating_mean)

print("\nAfter mean imputation:")
print(df_mean[['Name', 'Age', 'Salary', 'Experience', 'Rating']])

# 2.4 MEDIAN IMPUTATION
print("\n\n2.4 MEDIAN IMPUTATION")
print("Method: Replace missing values with the median of the column")
print("Best for: Numerical data with outliers or skewed distribution")

df_median = df_original.copy()
# Calculate median for numerical columns
age_median = df_median['Age'].median()
salary_median = df_median['Salary'].median()
experience_median = df_median['Experience'].median()
rating_median = df_median['Rating'].median()

print(f"Age median: {age_median:.2f}")
print(f"Salary median: {salary_median:.2f}")
print(f"Experience median: {experience_median:.2f}")
print(f"Rating median: {rating_median:.2f}")

# Fill missing values with median
df_median['Age'] = df_median['Age'].fillna(age_median)
df_median['Salary'] = df_median['Salary'].fillna(salary_median)
df_median['Experience'] = df_median['Experience'].fillna(experience_median)
df_median['Rating'] = df_median['Rating'].fillna(rating_median)

print("\nAfter median imputation:")
print(df_median[['Name', 'Age', 'Salary', 'Experience', 'Rating']])

# 2.5 MODE IMPUTATION
print("\n\n2.5 MODE IMPUTATION")
print("Method: Replace missing values with the most frequent value (mode)")
print("Best for: Categorical data")

df_mode = df_original.copy()
# Calculate mode for categorical columns
department_mode = df_mode['Department'].mode()[0]  # mode() returns a Series, take first value
print(f"Department mode: {department_mode}")

# Fill missing values with mode
df_mode['Department'] = df_mode['Department'].fillna(department_mode)

print("\nAfter mode imputation:")
print(df_mode[['Name', 'Department']])

# =============================================================================
# 3. ADVANCED IMPUTATION METHODS
# =============================================================================

print("\n\n3. ADVANCED IMPUTATION METHODS")
print("-" * 50)

# 3.1 FORWARD FILL (FFILL)
print("\n3.1 FORWARD FILL (FFILL)")
print("Method: Use the last valid observation to fill missing values")
print("Best for: Time series data where values tend to persist")

df_ffill = df_original.copy()
df_ffill_sorted = df_ffill.sort_values('Name')  # Sort for demonstration
df_ffill_sorted['Age'] = df_ffill_sorted['Age'].fillna(method='ffill')
df_ffill_sorted['Salary'] = df_ffill_sorted['Salary'].fillna(method='ffill')

print("After forward fill:")
print(df_ffill_sorted[['Name', 'Age', 'Salary']])

# 3.2 BACKWARD FILL (BFILL)
print("\n\n3.2 BACKWARD FILL (BFILL)")
print("Method: Use the next valid observation to fill missing values")

df_bfill = df_original.copy()
df_bfill_sorted = df_bfill.sort_values('Name')
df_bfill_sorted['Age'] = df_bfill_sorted['Age'].fillna(method='bfill')
df_bfill_sorted['Salary'] = df_bfill_sorted['Salary'].fillna(method='bfill')

print("After backward fill:")
print(df_bfill_sorted[['Name', 'Age', 'Salary']])

# 3.3 INTERPOLATION
print("\n\n3.3 INTERPOLATION")
print("Method: Estimate missing values based on other values")
print("Best for: Numerical data with trends or patterns")

df_interp = df_original.copy()
# Sort by a logical order for interpolation
df_interp_sorted = df_interp.sort_values('Name')

# Linear interpolation
df_interp_sorted['Age'] = df_interp_sorted['Age'].interpolate(method='linear')
df_interp_sorted['Salary'] = df_interp_sorted['Salary'].interpolate(method='linear')
df_interp_sorted['Experience'] = df_interp_sorted['Experience'].interpolate(method='linear')

print("After linear interpolation:")
print(df_interp_sorted[['Name', 'Age', 'Salary', 'Experience']])

# 3.4 GROUP-BASED IMPUTATION
print("\n\n3.4 GROUP-BASED IMPUTATION")
print("Method: Fill missing values based on groups/categories")
print("Best for: When missing values should depend on category")

df_group = df_original.copy()

# Fill missing salary based on department mean
df_group['Salary'] = df_group.groupby('Department')['Salary'].transform(
    lambda x: x.fillna(x.mean())
)

# Fill remaining missing values with overall mean
df_group['Salary'] = df_group['Salary'].fillna(df_group['Salary'].mean())

print("After group-based imputation (Salary by Department):")
print(df_group[['Name', 'Department', 'Salary']])

# =============================================================================
# 4. SCIKIT-LEARN IMPUTATION METHODS
# =============================================================================

print("\n\n4. SCIKIT-LEARN IMPUTATION METHODS")
print("-" * 50)

# Prepare numerical data for sklearn imputers
numerical_cols = ['Age', 'Salary', 'Experience', 'Rating']
df_numerical = df_original[numerical_cols].copy()

# 4.1 SIMPLE IMPUTER
print("\n4.1 SIMPLE IMPUTER")
print("Method: Basic imputation strategies using sklearn")

# Mean imputation using SimpleImputer
imputer_mean = SimpleImputer(strategy='mean')
df_sklearn_mean = pd.DataFrame(
    imputer_mean.fit_transform(df_numerical),
    columns=numerical_cols
)
print("SimpleImputer with mean strategy:")
print(df_sklearn_mean)

# Median imputation using SimpleImputer
imputer_median = SimpleImputer(strategy='median')
df_sklearn_median = pd.DataFrame(
    imputer_median.fit_transform(df_numerical),
    columns=numerical_cols
)
print("\nSimpleImputer with median strategy:")
print(df_sklearn_median)

# Most frequent imputation using SimpleImputer
imputer_frequent = SimpleImputer(strategy='most_frequent')
df_sklearn_frequent = pd.DataFrame(
    imputer_frequent.fit_transform(df_numerical),
    columns=numerical_cols
)
print("\nSimpleImputer with most_frequent strategy:")
print(df_sklearn_frequent)

# 4.2 KNN IMPUTER
print("\n\n4.2 KNN IMPUTER")
print("Method: Use K-Nearest Neighbors to impute missing values")
print("Best for: When similar records should have similar values")

knn_imputer = KNNImputer(n_neighbors=3)  # Use 3 nearest neighbors
df_knn = pd.DataFrame(
    knn_imputer.fit_transform(df_numerical),
    columns=numerical_cols
)
print("KNN Imputer (k=3):")
print(df_knn)

# 4.3 ITERATIVE IMPUTER
print("\n\n4.3 ITERATIVE IMPUTER")
print("Method: Model each feature with missing values as a function of other features")
print("Best for: When features are correlated")

iterative_imputer = IterativeImputer(random_state=42, max_iter=10)
df_iterative = pd.DataFrame(
    iterative_imputer.fit_transform(df_numerical),
    columns=numerical_cols
)
print("Iterative Imputer:")
print(df_iterative)

# =============================================================================
# 5. COMPARISON OF METHODS
# =============================================================================

print("\n\n5. COMPARISON OF IMPUTATION METHODS")
print("-" * 50)

# Create a comparison dataframe
comparison_data = {
    'Original': df_original['Age'].tolist(),
    'Mean': df_mean['Age'].tolist(),
    'Median': df_median['Age'].tolist(),
    'Forward Fill': df_ffill_sorted['Age'].tolist(),
    'Interpolation': df_interp_sorted['Age'].tolist(),
    'KNN': df_knn['Age'].tolist(),
    'Iterative': df_iterative['Age'].tolist()
}

comparison_df = pd.DataFrame(comparison_data, index=df_original['Name'])
print("Comparison of Age imputation methods:")
print(comparison_df.round(2))

# =============================================================================
# 6. BEST PRACTICES AND RECOMMENDATIONS
# =============================================================================

print("\n\n6. BEST PRACTICES AND RECOMMENDATIONS")
print("-" * 50)

recommendations = """
CHOOSING THE RIGHT IMPUTATION METHOD:

1. UNDERSTAND YOUR DATA:
   - Check the pattern of missing values (random vs systematic)
   - Understand the data distribution
   - Consider the relationship between features

2. METHOD SELECTION GUIDE:
   
   For NUMERICAL data:
   - Normal distribution → Mean imputation
   - Skewed distribution → Median imputation
   - Time series → Forward/Backward fill
   - Correlated features → KNN or Iterative imputation
   
   For CATEGORICAL data:
   - Mode imputation for most frequent category
   - Domain-specific constants (e.g., 'Unknown', 'Other')
   
   For TIME SERIES:
   - Forward fill for persistent values
   - Interpolation for trending data
   - Seasonal decomposition for complex patterns

3. ADVANCED CONSIDERATIONS:
   - Use multiple imputation for uncertainty quantification
   - Consider domain knowledge when choosing constants
   - Validate imputation quality with cross-validation
   - Document your imputation strategy for reproducibility

4. THINGS TO AVOID:
   - Don't use mean imputation for highly skewed data
   - Don't use forward fill for non-sequential data
   - Don't ignore the reason why data is missing
   - Don't impute without understanding the impact on analysis

5. VALIDATION:
   - Compare distributions before and after imputation
   - Check if imputation introduces bias
   - Use holdout validation to test imputation quality
"""

print(recommendations)

print("\n" + "=" * 60)
print("END OF NULL IMPUTATION DEMONSTRATION")
print("=" * 60)

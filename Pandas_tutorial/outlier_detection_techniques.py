"""
OUTLIER DETECTION AND HANDLING TECHNIQUES IN PANDAS
===================================================

This file demonstrates various methods for detecting, analyzing, and handling 
outliers in pandas DataFrames. Each technique is explained with comments and 
practical examples.

Author: Learning Pandas
Date: January 2026
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# Set up plotting
plt.style.use('default')
sns.set_palette("husl")

print("=" * 60)
print("OUTLIER DETECTION AND HANDLING TECHNIQUES")
print("=" * 60)

# =============================================================================
# 1. CREATING SAMPLE DATA WITH OUTLIERS
# =============================================================================

print("\n1. CREATING SAMPLE DATASET WITH OUTLIERS")
print("-" * 50)

# Set random seed for reproducibility
np.random.seed(42)

# Create a sample dataset with intentional outliers
n_samples = 100

# Normal data
normal_ages = np.random.normal(35, 8, n_samples-10)  # Mean 35, std 8
normal_salaries = np.random.normal(60000, 15000, n_samples-10)  # Mean 60k, std 15k
normal_experience = np.random.normal(8, 3, n_samples-10)  # Mean 8 years, std 3

# Add outliers
outlier_ages = [150, 5, 200, -10, 95]  # Impossible/extreme ages
outlier_salaries = [500000, 1000000, 5, -50000, 800000]  # Extreme salaries
outlier_experience = [50, -5, 60, 0.1, 45]  # Extreme experience values

# Combine normal and outlier data
ages = np.concatenate([normal_ages, outlier_ages])
salaries = np.concatenate([normal_salaries, outlier_salaries])
experience = np.concatenate([normal_experience, outlier_experience])

# Create additional features
performance_scores = np.random.normal(75, 10, n_samples)
# Add some extreme performance scores
performance_scores[-3:] = [10, 150, -20]

# Create department data
departments = np.random.choice(['IT', 'HR', 'Finance', 'Marketing', 'Sales'], n_samples)

# Create the DataFrame
data = {
    'EmployeeID': range(1001, 1001 + n_samples),
    'Age': ages,
    'Salary': salaries,
    'Experience': experience,
    'Performance_Score': performance_scores,
    'Department': departments,
    'Hours_Worked_Weekly': np.random.normal(40, 5, n_samples)
}

# Add some extreme hours worked
data['Hours_Worked_Weekly'][-5:] = [100, 5, 120, 2, 90]

df_original = pd.DataFrame(data)

print("Dataset with outliers:")
print(df_original.head(10))
print(f"\nDataset shape: {df_original.shape}")
print(f"\nBasic statistics:")
print(df_original.describe())

# =============================================================================
# 2. STATISTICAL OUTLIER DETECTION METHODS
# =============================================================================

print("\n\n2. STATISTICAL OUTLIER DETECTION METHODS")
print("-" * 50)

# 2.1 Z-SCORE METHOD
print("\n2.1 Z-SCORE METHOD")
print("Method: Identify outliers based on standard deviations from mean")
print("Rule: |z-score| > threshold (commonly 2 or 3)")

def detect_outliers_zscore(df, columns, threshold=3):
    """Detect outliers using Z-score method"""
    outliers_dict = {}
    
    for col in columns:
        if df[col].dtype in ['int64', 'float64']:
            z_scores = np.abs(stats.zscore(df[col]))
            outliers = df[z_scores > threshold]
            outliers_dict[col] = {
                'count': len(outliers),
                'indices': outliers.index.tolist(),
                'values': outliers[col].tolist(),
                'z_scores': z_scores[z_scores > threshold].tolist()
            }
    
    return outliers_dict

# Apply Z-score method
numerical_cols = ['Age', 'Salary', 'Experience', 'Performance_Score', 'Hours_Worked_Weekly']
zscore_outliers = detect_outliers_zscore(df_original, numerical_cols, threshold=3)

print("Z-score outliers (threshold=3):")
for col, info in zscore_outliers.items():
    print(f"\n{col}:")
    print(f"  Outliers found: {info['count']}")
    if info['count'] > 0:
        print(f"  Values: {[round(v, 2) if isinstance(v, float) else v for v in info['values'][:5]}")
        print(f"  Z-scores: {[round(z, 2) for z in info['z_scores'][:5]]}")

# 2.2 INTERQUARTILE RANGE (IQR) METHOD
print("\n\n2.2 INTERQUARTILE RANGE (IQR) METHOD")
print("Method: Identify outliers beyond Q1 - 1.5*IQR and Q3 + 1.5*IQR")
print("More robust to extreme values than Z-score")

def detect_outliers_iqr(df, columns, multiplier=1.5):
    """Detect outliers using IQR method"""
    outliers_dict = {}
    
    for col in columns:
        if df[col].dtype in ['int64', 'float64']:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            
            lower_bound = Q1 - multiplier * IQR
            upper_bound = Q3 + multiplier * IQR
            
            outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
            
            outliers_dict[col] = {
                'count': len(outliers),
                'lower_bound': lower_bound,
                'upper_bound': upper_bound,
                'Q1': Q1,
                'Q3': Q3,
                'IQR': IQR,
                'outlier_values': outliers[col].tolist(),
                'outlier_indices': outliers.index.tolist()
            }
    
    return outliers_dict

# Apply IQR method
iqr_outliers = detect_outliers_iqr(df_original, numerical_cols)

print("IQR outliers (multiplier=1.5):")
for col, info in iqr_outliers.items():
    print(f"\n{col}:")
    print(f"  Outliers found: {info['count']}")
    print(f"  Valid range: [{info['lower_bound']:.2f}, {info['upper_bound']:.2f}]")
    print(f"  Q1: {info['Q1']:.2f}, Q3: {info['Q3']:.2f}, IQR: {info['IQR']:.2f}")
    if info['count'] > 0:
        values = [round(v, 2) if isinstance(v, float) else v for v in info['outlier_values'][:5]]
        print(f"  Outlier values: {values}")

# 2.3 MODIFIED Z-SCORE (MEDIAN ABSOLUTE DEVIATION)
print("\n\n2.3 MODIFIED Z-SCORE (MEDIAN ABSOLUTE DEVIATION)")
print("Method: More robust alternative to Z-score using median")
print("Less sensitive to extreme outliers")

def detect_outliers_modified_zscore(df, columns, threshold=3.5):
    """Detect outliers using Modified Z-score (MAD)"""
    outliers_dict = {}
    
    for col in columns:
        if df[col].dtype in ['int64', 'float64']:
            median = df[col].median()
            mad = np.median(np.abs(df[col] - median))
            
            # Calculate modified z-scores
            modified_z_scores = 0.6745 * (df[col] - median) / mad
            
            outliers = df[np.abs(modified_z_scores) > threshold]
            
            outliers_dict[col] = {
                'count': len(outliers),
                'median': median,
                'mad': mad,
                'outlier_indices': outliers.index.tolist(),
                'outlier_values': outliers[col].tolist(),
                'modified_z_scores': modified_z_scores[np.abs(modified_z_scores) > threshold].tolist()
            }
    
    return outliers_dict

# Apply Modified Z-score method
mad_outliers = detect_outliers_modified_zscore(df_original, numerical_cols)

print("Modified Z-score outliers (threshold=3.5):")
for col, info in mad_outliers.items():
    print(f"\n{col}:")
    print(f"  Outliers found: {info['count']}")
    print(f"  Median: {info['median']:.2f}, MAD: {info['mad']:.2f}")
    if info['count'] > 0:
        values = [round(v, 2) if isinstance(v, float) else v for v in info['outlier_values'][:5]]
        scores = [round(s, 2) for s in info['modified_z_scores'][:5]]
        print(f"  Outlier values: {values}")
        print(f"  Modified Z-scores: {scores}")

# =============================================================================
# 3. VISUALIZATION-BASED OUTLIER DETECTION
# =============================================================================

print("\n\n3. VISUALIZATION-BASED OUTLIER DETECTION")
print("-" * 50)

print("\n3.1 BOX PLOTS FOR OUTLIER VISUALIZATION")
print("Method: Visual identification of outliers using box plots")

def create_boxplots(df, columns, figsize=(15, 10)):
    """Create box plots for outlier visualization"""
    n_cols = len(columns)
    n_rows = (n_cols + 2) // 3  # 3 columns per row
    
    fig, axes = plt.subplots(n_rows, 3, figsize=figsize)
    axes = axes.flatten() if n_rows > 1 else [axes] if n_rows == 1 else axes
    
    for i, col in enumerate(columns):
        if i < len(axes):
            df.boxplot(column=col, ax=axes[i])
            axes[i].set_title(f'{col} - Box Plot')
            axes[i].grid(True, alpha=0.3)
    
    # Hide unused subplots
    for i in range(len(columns), len(axes)):
        axes[i].set_visible(False)
    
    plt.tight_layout()
    plt.savefig('outliers_boxplots.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("Box plots saved as 'outliers_boxplots.png'")

# Create box plots (commented out to avoid display issues in some environments)
# create_boxplots(df_original, numerical_cols)

print("Box plots help identify:")
print("- Points beyond whiskers are potential outliers")
print("- Distribution shape and skewness")
print("- Comparative outlier patterns across variables")

# 3.2 SCATTER PLOTS FOR MULTIVARIATE OUTLIERS
print("\n\n3.2 SCATTER PLOTS FOR MULTIVARIATE OUTLIERS")
print("Method: Identify outliers in relationships between variables")

def analyze_bivariate_outliers(df, col1, col2):
    """Analyze outliers in bivariate relationships"""
    
    # Calculate correlation
    correlation = df[col1].corr(df[col2])
    
    # Identify potential outliers using both variables
    z1 = np.abs(stats.zscore(df[col1]))
    z2 = np.abs(stats.zscore(df[col2]))
    
    # Points that are outliers in either dimension
    outliers_either = df[(z1 > 2) | (z2 > 2)]
    
    # Points that are outliers in both dimensions
    outliers_both = df[(z1 > 2) & (z2 > 2)]
    
    print(f"\nBivariate analysis: {col1} vs {col2}")
    print(f"Correlation: {correlation:.3f}")
    print(f"Outliers in either dimension: {len(outliers_either)}")
    print(f"Outliers in both dimensions: {len(outliers_both)}")
    
    return outliers_either, outliers_both

# Analyze key relationships
bivariate_outliers = {}
relationships = [('Age', 'Experience'), ('Salary', 'Experience'), ('Performance_Score', 'Salary')]

for col1, col2 in relationships:
    either, both = analyze_bivariate_outliers(df_original, col1, col2)
    bivariate_outliers[f"{col1}_vs_{col2}"] = {'either': either, 'both': both}

# =============================================================================
# 4. ADVANCED OUTLIER DETECTION METHODS
# =============================================================================

print("\n\n4. ADVANCED OUTLIER DETECTION METHODS")
print("-" * 50)

# 4.1 ISOLATION FOREST
print("\n4.1 ISOLATION FOREST")
print("Method: Machine learning approach for anomaly detection")
print("Works well for high-dimensional data and doesn't assume data distribution")

def detect_outliers_isolation_forest(df, columns, contamination=0.1):
    """Detect outliers using Isolation Forest"""
    
    # Prepare data
    data_for_analysis = df[columns].copy()
    
    # Handle any missing values
    data_for_analysis = data_for_analysis.fillna(data_for_analysis.mean())
    
    # Standardize the data
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data_for_analysis)
    
    # Apply Isolation Forest
    iso_forest = IsolationForest(contamination=contamination, random_state=42)
    outlier_labels = iso_forest.fit_predict(data_scaled)
    
    # Get outliers (labeled as -1)
    outliers = df[outlier_labels == -1]
    
    return outliers, outlier_labels

# Apply Isolation Forest
iso_outliers, iso_labels = detect_outliers_isolation_forest(df_original, numerical_cols, contamination=0.05)

print(f"Isolation Forest outliers (contamination=0.05):")
print(f"Outliers detected: {len(iso_outliers)}")
print(f"Outlier indices: {iso_outliers.index.tolist()}")
print("\nOutlier records:")
print(iso_outliers[['EmployeeID', 'Age', 'Salary', 'Experience', 'Performance_Score']].head())

# 4.2 LOCAL OUTLIER FACTOR (LOF)
print("\n\n4.2 LOCAL OUTLIER FACTOR (LOF)")
print("Method: Density-based outlier detection")
print("Identifies outliers based on local density compared to neighbors")

def detect_outliers_lof(df, columns, n_neighbors=20, contamination=0.1):
    """Detect outliers using Local Outlier Factor"""
    
    # Prepare data
    data_for_analysis = df[columns].copy()
    data_for_analysis = data_for_analysis.fillna(data_for_analysis.mean())
    
    # Standardize the data
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data_for_analysis)
    
    # Apply LOF
    lof = LocalOutlierFactor(n_neighbors=n_neighbors, contamination=contamination)
    outlier_labels = lof.fit_predict(data_scaled)
    
    # Get outlier scores
    outlier_scores = lof.negative_outlier_factor_
    
    # Get outliers (labeled as -1)
    outliers = df[outlier_labels == -1]
    
    return outliers, outlier_labels, outlier_scores

# Apply LOF
lof_outliers, lof_labels, lof_scores = detect_outliers_lof(df_original, numerical_cols, contamination=0.05)

print(f"Local Outlier Factor outliers (contamination=0.05):")
print(f"Outliers detected: {len(lof_outliers)}")
print(f"Outlier indices: {lof_outliers.index.tolist()}")

# Show outliers with their LOF scores
if len(lof_outliers) > 0:
    outlier_analysis = lof_outliers.copy()
    outlier_analysis['LOF_Score'] = lof_scores[lof_labels == -1]
    print("\nOutliers with LOF scores (more negative = more outlying):")
    print(outlier_analysis[['EmployeeID', 'Age', 'Salary', 'Experience', 'LOF_Score']].head())

# 4.3 DOMAIN-SPECIFIC OUTLIER DETECTION
print("\n\n4.3 DOMAIN-SPECIFIC OUTLIER DETECTION")
print("Method: Business rule-based outlier detection")
print("Uses domain knowledge to identify impossible or suspicious values")

def detect_domain_outliers(df):
    """Detect outliers based on business rules and domain knowledge"""
    
    outliers = {
        'impossible_age': df[(df['Age'] < 16) | (df['Age'] > 80)],
        'negative_salary': df[df['Salary'] < 0],
        'excessive_salary': df[df['Salary'] > 300000],  # Assuming max reasonable salary
        'negative_experience': df[df['Experience'] < 0],
        'impossible_experience': df[df['Experience'] > df['Age'] - 15],  # Started working before age 15
        'extreme_hours': df[(df['Hours_Worked_Weekly'] < 10) | (df['Hours_Worked_Weekly'] > 80)],
        'impossible_performance': df[(df['Performance_Score'] < 0) | (df['Performance_Score'] > 100)]
    }
    
    return outliers

# Apply domain-specific detection
domain_outliers = detect_domain_outliers(df_original)

print("Domain-specific outliers:")
for rule, outliers in domain_outliers.items():
    if len(outliers) > 0:
        print(f"\n{rule.replace('_', ' ').title()}:")
        print(f"  Count: {len(outliers)}")
        print(f"  Employee IDs: {outliers['EmployeeID'].tolist()}")
        
        # Show specific values for the relevant column
        if 'age' in rule:
            print(f"  Ages: {outliers['Age'].tolist()}")
        elif 'salary' in rule:
            print(f"  Salaries: {[f'${s:,.0f}' for s in outliers['Salary'].tolist()]}")
        elif 'experience' in rule:
            print(f"  Experience: {outliers['Experience'].tolist()}")
        elif 'hours' in rule:
            print(f"  Hours: {outliers['Hours_Worked_Weekly'].tolist()}")
        elif 'performance' in rule:
            print(f"  Scores: {outliers['Performance_Score'].tolist()}")

# =============================================================================
# 5. OUTLIER HANDLING STRATEGIES
# =============================================================================

print("\n\n5. OUTLIER HANDLING STRATEGIES")
print("-" * 50)

# 5.1 REMOVAL STRATEGIES
print("\n5.1 OUTLIER REMOVAL STRATEGIES")
print("Method: Remove outliers from the dataset")

def remove_outliers_iqr(df, columns, multiplier=1.5):
    """Remove outliers using IQR method"""
    df_clean = df.copy()
    removed_count = 0
    
    for col in columns:
        if df_clean[col].dtype in ['int64', 'float64']:
            Q1 = df_clean[col].quantile(0.25)
            Q3 = df_clean[col].quantile(0.75)
            IQR = Q3 - Q1
            
            lower_bound = Q1 - multiplier * IQR
            upper_bound = Q3 + multiplier * IQR
            
            before_count = len(df_clean)
            df_clean = df_clean[(df_clean[col] >= lower_bound) & (df_clean[col] <= upper_bound)]
            after_count = len(df_clean)
            
            removed_count += (before_count - after_count)
    
    return df_clean, removed_count

# Remove outliers using IQR
df_no_outliers, removed_count = remove_outliers_iqr(df_original, numerical_cols)

print(f"Outlier removal using IQR method:")
print(f"Original dataset: {len(df_original)} records")
print(f"After removal: {len(df_no_outliers)} records")
print(f"Records removed: {removed_count}")
print(f"Percentage removed: {(removed_count/len(df_original))*100:.1f}%")

# 5.2 CAPPING/WINSORIZING
print("\n\n5.2 CAPPING/WINSORIZING")
print("Method: Cap outliers at specific percentiles instead of removing")

def cap_outliers(df, columns, lower_percentile=5, upper_percentile=95):
    """Cap outliers at specified percentiles"""
    df_capped = df.copy()
    
    for col in columns:
        if df_capped[col].dtype in ['int64', 'float64']:
            lower_cap = df_capped[col].quantile(lower_percentile/100)
            upper_cap = df_capped[col].quantile(upper_percentile/100)
            
            # Count values that will be capped
            lower_capped = (df_capped[col] < lower_cap).sum()
            upper_capped = (df_capped[col] > upper_cap).sum()
            
            # Apply capping
            df_capped[col] = df_capped[col].clip(lower=lower_cap, upper=upper_cap)
            
            print(f"{col}:")
            print(f"  Lower cap ({lower_percentile}th percentile): {lower_cap:.2f} ({lower_capped} values capped)")
            print(f"  Upper cap ({upper_percentile}th percentile): {upper_cap:.2f} ({upper_capped} values capped)")
    
    return df_capped

# Apply capping
df_capped = cap_outliers(df_original, numerical_cols, lower_percentile=5, upper_percentile=95)

print(f"\nDataset shape remains: {df_capped.shape}")

# 5.3 TRANSFORMATION METHODS
print("\n\n5.3 TRANSFORMATION METHODS")
print("Method: Transform data to reduce impact of outliers")

def apply_transformations(df, columns):
    """Apply various transformations to reduce outlier impact"""
    df_transformed = df.copy()
    
    transformations = {}
    
    for col in columns:
        if df_transformed[col].dtype in ['int64', 'float64'] and (df_transformed[col] > 0).all():
            
            # Log transformation (only for positive values)
            if (df_transformed[col] > 0).all():
                df_transformed[f'{col}_log'] = np.log(df_transformed[col])
                transformations[f'{col}_log'] = 'Log transformation'
            
            # Square root transformation
            if (df_transformed[col] >= 0).all():
                df_transformed[f'{col}_sqrt'] = np.sqrt(df_transformed[col])
                transformations[f'{col}_sqrt'] = 'Square root transformation'
    
    return df_transformed, transformations

# Apply transformations (only to positive columns)
positive_cols = ['Salary', 'Experience', 'Performance_Score', 'Hours_Worked_Weekly']
df_transformed, transformations = apply_transformations(df_original, positive_cols)

print("Applied transformations:")
for col, method in transformations.items():
    print(f"  {col}: {method}")

# Compare distributions before and after transformation
print(f"\nOriginal Salary statistics:")
print(df_original['Salary'].describe())

if 'Salary_log' in df_transformed.columns:
    print(f"\nLog-transformed Salary statistics:")
    print(df_transformed['Salary_log'].describe())

# 5.4 IMPUTATION METHODS
print("\n\n5.4 OUTLIER IMPUTATION METHODS")
print("Method: Replace outliers with more reasonable values")

def impute_outliers_iqr(df, columns, method='median', multiplier=1.5):
    """Replace outliers with imputed values"""
    df_imputed = df.copy()
    
    for col in columns:
        if df_imputed[col].dtype in ['int64', 'float64']:
            Q1 = df_imputed[col].quantile(0.25)
            Q3 = df_imputed[col].quantile(0.75)
            IQR = Q3 - Q1
            
            lower_bound = Q1 - multiplier * IQR
            upper_bound = Q3 + multiplier * IQR
            
            # Identify outliers
            outliers_mask = (df_imputed[col] < lower_bound) | (df_imputed[col] > upper_bound)
            outliers_count = outliers_mask.sum()
            
            if outliers_count > 0:
                # Choose imputation value
                if method == 'median':
                    impute_value = df_imputed[col].median()
                elif method == 'mean':
                    # Use mean of non-outlier values
                    impute_value = df_imputed[~outliers_mask][col].mean()
                elif method == 'boundary':
                    # Replace with boundary values
                    df_imputed.loc[df_imputed[col] < lower_bound, col] = lower_bound
                    df_imputed.loc[df_imputed[col] > upper_bound, col] = upper_bound
                    continue
                
                # Apply imputation
                if method != 'boundary':
                    df_imputed.loc[outliers_mask, col] = impute_value
                
                print(f"{col}: {outliers_count} outliers imputed with {method}")
    
    return df_imputed

# Apply different imputation methods
df_imputed_median = impute_outliers_iqr(df_original, numerical_cols, method='median')
df_imputed_boundary = impute_outliers_iqr(df_original, numerical_cols, method='boundary')

# =============================================================================
# 6. COMPARISON OF METHODS
# =============================================================================

print("\n\n6. COMPARISON OF OUTLIER DETECTION METHODS")
print("-" * 50)

# Create comparison of different methods
def compare_outlier_methods(df, columns):
    """Compare different outlier detection methods"""
    
    results = {}
    
    # Z-score method
    zscore_outliers = detect_outliers_zscore(df, columns, threshold=3)
    zscore_total = sum(info['count'] for info in zscore_outliers.values())
    results['Z-Score (threshold=3)'] = zscore_total
    
    # IQR method
    iqr_outliers = detect_outliers_iqr(df, columns, multiplier=1.5)
    iqr_total = sum(info['count'] for info in iqr_outliers.values())
    results['IQR (multiplier=1.5)'] = iqr_total
    
    # Modified Z-score
    mad_outliers = detect_outliers_modified_zscore(df, columns, threshold=3.5)
    mad_total = sum(info['count'] for info in mad_outliers.values())
    results['Modified Z-Score (threshold=3.5)'] = mad_total
    
    # Isolation Forest
    iso_outliers, _ = detect_outliers_isolation_forest(df, columns, contamination=0.05)
    results['Isolation Forest (contamination=0.05)'] = len(iso_outliers)
    
    # LOF
    lof_outliers, _, _ = detect_outliers_lof(df, columns, contamination=0.05)
    results['LOF (contamination=0.05)'] = len(lof_outliers)
    
    return results

comparison_results = compare_outlier_methods(df_original, numerical_cols)

print("Outliers detected by different methods:")
for method, count in comparison_results.items():
    print(f"  {method}: {count} outliers")

# =============================================================================
# 7. BEST PRACTICES AND RECOMMENDATIONS
# =============================================================================

print("\n\n7. BEST PRACTICES AND RECOMMENDATIONS")
print("-" * 50)

recommendations = """
OUTLIER DETECTION AND HANDLING BEST PRACTICES:

1. UNDERSTAND YOUR DATA FIRST:
   - Examine data distribution and business context
   - Distinguish between errors and legitimate extreme values
   - Consider domain-specific constraints and rules
   - Visualize data before applying any methods

2. DETECTION METHOD SELECTION:

   For NORMALLY DISTRIBUTED data:
   - Z-Score method (threshold: 2-3 standard deviations)
   - Good for: Symmetric distributions, known normal data
   
   For SKEWED or NON-NORMAL data:
   - IQR method (multiplier: 1.5-3.0)
   - Modified Z-Score (MAD) for robust detection
   - Good for: Any distribution shape
   
   For HIGH-DIMENSIONAL data:
   - Isolation Forest for multivariate outliers
   - Local Outlier Factor for density-based detection
   - Good for: Complex patterns, multiple variables
   
   For DOMAIN-SPECIFIC cases:
   - Business rule-based detection
   - Combine multiple methods for validation

3. HANDLING STRATEGIES:

   REMOVAL:
   ✓ When outliers are clearly errors or impossible values
   ✓ When you have sufficient data after removal
   ✗ Don't remove without understanding why they exist
   
   CAPPING/WINSORIZING:
   ✓ When you want to preserve all records
   ✓ For reducing impact while keeping information
   ✓ Good for machine learning models
   
   TRANSFORMATION:
   ✓ Log/sqrt transforms for right-skewed data
   ✓ When outliers follow a pattern
   ✓ For normalizing distributions
   
   IMPUTATION:
   ✓ When outliers are measurement errors
   ✓ Replace with median/mean of clean data
   ✓ Use boundary values for reasonable limits

4. VALIDATION AND QUALITY CHECKS:
   ✓ Always visualize data before and after treatment
   ✓ Check impact on statistical properties (mean, std, etc.)
   ✓ Validate business logic is preserved
   ✓ Consider multiple detection methods for confirmation
   ✓ Document decisions and thresholds used

5. COMMON SCENARIOS:

   FINANCIAL DATA:
   - Use domain rules (negative prices, impossible amounts)
   - Consider log transformation for monetary values
   - Be careful with legitimate high-value transactions
   
   SENSOR/IOT DATA:
   - Check for instrument malfunctions
   - Use time-based patterns for validation
   - Consider environmental factors
   
   CUSTOMER DATA:
   - Age, income, purchase amounts have natural bounds
   - Consider seasonal patterns and special events
   - Validate against external data sources

6. THINGS TO AVOID:
   ✗ Don't automatically remove all statistical outliers
   ✗ Don't use the same threshold for all variables
   ✗ Don't ignore the business impact of outlier treatment
   ✗ Don't apply methods without understanding assumptions
   ✗ Don't forget to document your outlier handling process

7. ADVANCED CONSIDERATIONS:
   - Use ensemble methods (combine multiple techniques)
   - Consider temporal patterns in time series data
   - Account for seasonal variations and trends
   - Use cross-validation to test outlier detection stability
   - Consider the downstream impact on analysis/modeling

8. DOCUMENTATION CHECKLIST:
   ✓ Method used and parameters chosen
   ✓ Number of outliers detected and handled
   ✓ Business justification for treatment approach
   ✓ Impact on data distribution and statistics
   ✓ Validation steps performed
"""

print(recommendations)

# =============================================================================
# 8. SUMMARY STATISTICS
# =============================================================================

print("\n\n8. SUMMARY OF OUTLIER TREATMENT IMPACT")
print("-" * 50)

# Create summary comparison
summary_stats = {
    'Dataset': ['Original', 'Outliers Removed (IQR)', 'Outliers Capped (5-95%)', 'Outliers Imputed (Median)'],
    'Records': [len(df_original), len(df_no_outliers), len(df_capped), len(df_imputed_median)],
    'Age_Mean': [
        df_original['Age'].mean(),
        df_no_outliers['Age'].mean(),
        df_capped['Age'].mean(),
        df_imputed_median['Age'].mean()
    ],
    'Salary_Mean': [
        df_original['Salary'].mean(),
        df_no_outliers['Salary'].mean(),
        df_capped['Salary'].mean(),
        df_imputed_median['Salary'].mean()
    ],
    'Age_Std': [
        df_original['Age'].std(),
        df_no_outliers['Age'].std(),
        df_capped['Age'].std(),
        df_imputed_median['Age'].std()
    ],
    'Salary_Std': [
        df_original['Salary'].std(),
        df_no_outliers['Salary'].std(),
        df_capped['Salary'].std(),
        df_imputed_median['Salary'].std()
    ]
}

summary_df = pd.DataFrame(summary_stats)
summary_df = summary_df.round(2)

print("Impact of different outlier handling methods:")
print(summary_df)

print(f"\nKey Insights:")
print(f"- Original dataset had extreme values affecting means and standard deviations")
print(f"- Removal method eliminated {len(df_original) - len(df_no_outliers)} records")
print(f"- Capping preserved all records while reducing extreme impact")
print(f"- Imputation maintained record count with reasonable value substitution")

print("\n" + "=" * 60)
print("END OF OUTLIER DETECTION AND HANDLING DEMONSTRATION")
print("=" * 60)

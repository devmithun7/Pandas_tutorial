"""
DUPLICATE HANDLING TECHNIQUES IN PANDAS
=======================================

This file demonstrates various methods for detecting, analyzing, and handling 
duplicate records in pandas DataFrames. Each technique is explained with 
comments and practical examples.

Author: Learning Pandas
Date: January 2026
"""

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

print("=" * 60)
print("DUPLICATE HANDLING TECHNIQUES DEMONSTRATION")
print("=" * 60)

# =============================================================================
# 1. CREATING SAMPLE DATA WITH DUPLICATES
# =============================================================================

print("\n1. CREATING SAMPLE DATASET WITH DUPLICATES")
print("-" * 50)

# Create a sample dataset with various types of duplicates
data = {
    'CustomerID': [1001, 1002, 1003, 1001, 1004, 1005, 1002, 1006, 1003, 1007, 1001],
    'Name': ['Alice Johnson', 'Bob Smith', 'Charlie Brown', 'Alice Johnson', 
             'Diana Prince', 'Eve Adams', 'Bob Smith', 'Frank Miller', 
             'Charlie Brown', 'Grace Lee', 'Alice Johnson'],
    'Email': ['alice@email.com', 'bob@email.com', 'charlie@email.com', 'alice@email.com',
              'diana@email.com', 'eve@email.com', 'bob@email.com', 'frank@email.com',
              'charlie@email.com', 'grace@email.com', 'alice.johnson@email.com'],
    'Phone': ['555-0101', '555-0102', '555-0103', '555-0101', 
              '555-0104', '555-0105', '555-0102', '555-0106',
              '555-0103', '555-0107', '555-0101'],
    'City': ['New York', 'Los Angeles', 'Chicago', 'New York',
             'Houston', 'Phoenix', 'Los Angeles', 'Philadelphia',
             'Chicago', 'San Antonio', 'New York'],
    'Purchase_Amount': [250.50, 180.75, 320.00, 275.25,
                       150.00, 420.50, 195.30, 380.75,
                       340.20, 210.80, 290.15],
    'Purchase_Date': ['2024-01-15', '2024-01-16', '2024-01-17', '2024-01-20',
                     '2024-01-18', '2024-01-19', '2024-01-21', '2024-01-22',
                     '2024-01-23', '2024-01-24', '2024-01-25']
}

df_original = pd.DataFrame(data)
df_original['Purchase_Date'] = pd.to_datetime(df_original['Purchase_Date'])

print("Original dataset with duplicates:")
print(df_original)
print(f"\nDataset shape: {df_original.shape}")

# =============================================================================
# 2. DETECTING DUPLICATES
# =============================================================================

print("\n\n2. DETECTING DUPLICATES")
print("-" * 50)

# 2.1 COMPLETE ROW DUPLICATES
print("\n2.1 DETECTING COMPLETE ROW DUPLICATES")
print("Method: Check if entire rows are identical")

# Check for complete duplicates
complete_duplicates = df_original.duplicated()
print(f"Complete row duplicates found: {complete_duplicates.sum()}")
print("Rows that are complete duplicates:")
print(df_original[complete_duplicates])

# Show all instances of complete duplicates (including first occurrence)
complete_duplicates_all = df_original.duplicated(keep=False)
print(f"\nAll instances of complete duplicates: {complete_duplicates_all.sum()}")
if complete_duplicates_all.sum() > 0:
    print(df_original[complete_duplicates_all].sort_values('CustomerID'))

# 2.2 DUPLICATES BASED ON SPECIFIC COLUMNS
print("\n\n2.2 DUPLICATES BASED ON SPECIFIC COLUMNS")
print("Method: Check for duplicates based on key columns")

# Check for duplicates based on CustomerID
customer_duplicates = df_original.duplicated(subset=['CustomerID'])
print(f"Duplicate CustomerIDs found: {customer_duplicates.sum()}")
print("Rows with duplicate CustomerIDs:")
print(df_original[customer_duplicates])

# Show all instances of CustomerID duplicates
customer_duplicates_all = df_original.duplicated(subset=['CustomerID'], keep=False)
print(f"\nAll instances of duplicate CustomerIDs: {customer_duplicates_all.sum()}")
print(df_original[customer_duplicates_all].sort_values('CustomerID'))

# Check for duplicates based on multiple columns
name_email_duplicates = df_original.duplicated(subset=['Name', 'Email'])
print(f"\nDuplicate Name+Email combinations: {name_email_duplicates.sum()}")
if name_email_duplicates.sum() > 0:
    print(df_original[name_email_duplicates])

# 2.3 ANALYZING DUPLICATE PATTERNS
print("\n\n2.3 ANALYZING DUPLICATE PATTERNS")
print("Method: Understand the nature and frequency of duplicates")

# Count duplicates by CustomerID
customer_counts = df_original['CustomerID'].value_counts()
print("Customer ID frequency:")
print(customer_counts)

# Find customers with multiple records
multiple_records = customer_counts[customer_counts > 1]
print(f"\nCustomers with multiple records: {len(multiple_records)}")
print(multiple_records)

# Analyze duplicate patterns
print("\nDetailed analysis of duplicate customers:")
for customer_id in multiple_records.index:
    customer_data = df_original[df_original['CustomerID'] == customer_id]
    print(f"\nCustomer {customer_id}:")
    print(customer_data[['CustomerID', 'Name', 'Email', 'Purchase_Amount', 'Purchase_Date']])

# =============================================================================
# 3. BASIC DUPLICATE REMOVAL METHODS
# =============================================================================

print("\n\n3. BASIC DUPLICATE REMOVAL METHODS")
print("-" * 50)

# 3.1 DROP ALL COMPLETE DUPLICATES
print("\n3.1 REMOVING COMPLETE ROW DUPLICATES")
print("Method: Remove rows that are completely identical")

df_no_complete_dups = df_original.drop_duplicates()
print(f"Original shape: {df_original.shape}")
print(f"After removing complete duplicates: {df_no_complete_dups.shape}")
print(f"Rows removed: {df_original.shape[0] - df_no_complete_dups.shape[0]}")

# 3.2 DROP DUPLICATES BASED ON SPECIFIC COLUMNS
print("\n\n3.2 REMOVING DUPLICATES BASED ON SPECIFIC COLUMNS")
print("Method: Remove duplicates based on key identifier columns")

# Remove duplicates based on CustomerID (keep first occurrence)
df_unique_customers_first = df_original.drop_duplicates(subset=['CustomerID'], keep='first')
print(f"Keeping first occurrence of each CustomerID:")
print(f"Shape after deduplication: {df_unique_customers_first.shape}")
print(df_unique_customers_first.sort_values('CustomerID'))

# Remove duplicates based on CustomerID (keep last occurrence)
df_unique_customers_last = df_original.drop_duplicates(subset=['CustomerID'], keep='last')
print(f"\nKeeping last occurrence of each CustomerID:")
print(f"Shape after deduplication: {df_unique_customers_last.shape}")
print(df_unique_customers_last.sort_values('CustomerID'))

# 3.3 DIFFERENT 'KEEP' STRATEGIES
print("\n\n3.3 DIFFERENT 'KEEP' STRATEGIES")
print("Method: Control which duplicate to keep")

print("Available 'keep' options:")
print("- 'first': Keep first occurrence (default)")
print("- 'last': Keep last occurrence") 
print("- False: Remove all duplicates (keep none)")

# Remove ALL duplicates (keep none)
df_remove_all_dups = df_original.drop_duplicates(subset=['CustomerID'], keep=False)
print(f"\nRemoving ALL duplicate CustomerIDs (keep=False):")
print(f"Shape after removing all duplicates: {df_remove_all_dups.shape}")
print(df_remove_all_dups)

# =============================================================================
# 4. ADVANCED DUPLICATE HANDLING STRATEGIES
# =============================================================================

print("\n\n4. ADVANCED DUPLICATE HANDLING STRATEGIES")
print("-" * 50)

# 4.1 CONDITIONAL DUPLICATE REMOVAL
print("\n4.1 CONDITIONAL DUPLICATE REMOVAL")
print("Method: Choose which duplicate to keep based on conditions")

# Keep the record with the highest purchase amount for each customer
df_conditional = df_original.copy()
df_conditional = df_conditional.sort_values(['CustomerID', 'Purchase_Amount'], ascending=[True, False])
df_max_purchase = df_conditional.drop_duplicates(subset=['CustomerID'], keep='first')

print("Keeping record with highest purchase amount for each customer:")
print(df_max_purchase.sort_values('CustomerID')[['CustomerID', 'Name', 'Purchase_Amount', 'Purchase_Date']])

# Keep the most recent purchase for each customer
df_recent = df_original.copy()
df_recent = df_recent.sort_values(['CustomerID', 'Purchase_Date'], ascending=[True, False])
df_most_recent = df_recent.drop_duplicates(subset=['CustomerID'], keep='first')

print("\nKeeping most recent purchase for each customer:")
print(df_most_recent.sort_values('CustomerID')[['CustomerID', 'Name', 'Purchase_Amount', 'Purchase_Date']])

# 4.2 AGGREGATING DUPLICATES
print("\n\n4.2 AGGREGATING DUPLICATES")
print("Method: Combine duplicate records instead of removing them")

# Aggregate purchases by customer
df_aggregated = df_original.groupby(['CustomerID', 'Name', 'Email', 'Phone', 'City']).agg({
    'Purchase_Amount': ['sum', 'mean', 'count'],
    'Purchase_Date': ['min', 'max']
}).round(2)

# Flatten column names
df_aggregated.columns = ['Total_Purchases', 'Avg_Purchase', 'Purchase_Count', 'First_Purchase', 'Last_Purchase']
df_aggregated = df_aggregated.reset_index()

print("Aggregated customer data:")
print(df_aggregated)

# 4.3 FUZZY DUPLICATE DETECTION
print("\n\n4.3 FUZZY DUPLICATE DETECTION")
print("Method: Find similar but not identical records")

# Create a dataset with slight variations
fuzzy_data = {
    'Name': ['John Smith', 'Jon Smith', 'John Smyth', 'Jane Doe', 'Jane Do', 'Bob Johnson'],
    'Email': ['john@email.com', 'jon@email.com', 'john@email.com', 
              'jane@email.com', 'jane@email.com', 'bob@email.com'],
    'Phone': ['555-1234', '555-1234', '555-1235', '555-5678', '555-5678', '555-9999']
}

df_fuzzy = pd.DataFrame(fuzzy_data)
print("Dataset with potential fuzzy duplicates:")
print(df_fuzzy)

# Simple fuzzy matching using string similarity
def similar_names(df, column, threshold=0.8):
    """Find potentially similar names"""
    from difflib import SequenceMatcher
    
    similar_pairs = []
    names = df[column].tolist()
    
    for i, name1 in enumerate(names):
        for j, name2 in enumerate(names[i+1:], i+1):
            similarity = SequenceMatcher(None, name1.lower(), name2.lower()).ratio()
            if similarity >= threshold:
                similar_pairs.append((i, j, name1, name2, similarity))
    
    return similar_pairs

try:
    similar_pairs = similar_names(df_fuzzy, 'Name', threshold=0.7)
    print(f"\nPotentially similar names (similarity >= 70%):")
    for pair in similar_pairs:
        print(f"Row {pair[0]} '{pair[2]}' vs Row {pair[1]} '{pair[3]}' - Similarity: {pair[4]:.2%}")
except ImportError:
    print("\nNote: difflib not available for fuzzy matching demonstration")

# =============================================================================
# 5. DUPLICATE VALIDATION AND QUALITY CHECKS
# =============================================================================

print("\n\n5. DUPLICATE VALIDATION AND QUALITY CHECKS")
print("-" * 50)

# 5.1 VALIDATE DEDUPLICATION RESULTS
print("\n5.1 VALIDATING DEDUPLICATION RESULTS")
print("Method: Ensure deduplication worked correctly")

def validate_deduplication(original_df, deduplicated_df, key_columns):
    """Validate that deduplication was successful"""
    
    print(f"Original records: {len(original_df)}")
    print(f"After deduplication: {len(deduplicated_df)}")
    print(f"Records removed: {len(original_df) - len(deduplicated_df)}")
    
    # Check for remaining duplicates
    remaining_dups = deduplicated_df.duplicated(subset=key_columns).sum()
    print(f"Remaining duplicates in key columns: {remaining_dups}")
    
    # Check unique values
    original_unique = original_df[key_columns[0]].nunique()
    final_unique = deduplicated_df[key_columns[0]].nunique()
    print(f"Unique {key_columns[0]} values - Original: {original_unique}, Final: {final_unique}")
    
    return remaining_dups == 0

print("Validation of CustomerID deduplication:")
is_valid = validate_deduplication(df_original, df_unique_customers_first, ['CustomerID'])
print(f"Deduplication successful: {is_valid}")

# 5.2 DATA QUALITY IMPACT ANALYSIS
print("\n\n5.2 DATA QUALITY IMPACT ANALYSIS")
print("Method: Analyze the impact of duplicate removal on data quality")

def analyze_deduplication_impact(original_df, deduplicated_df):
    """Analyze the impact of deduplication on data distribution"""
    
    print("Impact Analysis:")
    print("-" * 30)
    
    # Numerical columns impact
    numerical_cols = original_df.select_dtypes(include=[np.number]).columns
    
    for col in numerical_cols:
        orig_mean = original_df[col].mean()
        dedup_mean = deduplicated_df[col].mean()
        change = ((dedup_mean - orig_mean) / orig_mean) * 100
        
        print(f"{col}:")
        print(f"  Original mean: {orig_mean:.2f}")
        print(f"  Deduplicated mean: {dedup_mean:.2f}")
        print(f"  Change: {change:+.2f}%")
    
    # Categorical distribution impact
    categorical_cols = original_df.select_dtypes(include=['object']).columns
    
    for col in categorical_cols:
        if col in ['Name', 'Email']:  # Skip unique identifiers
            continue
            
        print(f"\n{col} distribution change:")
        orig_dist = original_df[col].value_counts(normalize=True)
        dedup_dist = deduplicated_df[col].value_counts(normalize=True)
        
        for category in orig_dist.index:
            orig_pct = orig_dist.get(category, 0) * 100
            dedup_pct = dedup_dist.get(category, 0) * 100
            change = dedup_pct - orig_pct
            print(f"  {category}: {orig_pct:.1f}% → {dedup_pct:.1f}% ({change:+.1f}%)")

analyze_deduplication_impact(df_original, df_unique_customers_first)

# =============================================================================
# 6. SPECIALIZED DUPLICATE SCENARIOS
# =============================================================================

print("\n\n6. SPECIALIZED DUPLICATE SCENARIOS")
print("-" * 50)

# 6.1 TIME-BASED DUPLICATES
print("\n6.1 HANDLING TIME-BASED DUPLICATES")
print("Method: Deal with duplicates in time series data")

# Create time series data with duplicates
time_data = {
    'Timestamp': pd.date_range('2024-01-01', periods=10, freq='H'),
    'Sensor_ID': ['A001', 'A001', 'A001', 'B002', 'B002', 'A001', 'C003', 'C003', 'B002', 'A001'],
    'Temperature': [23.5, 23.5, 24.1, 22.8, 22.8, 24.3, 25.2, 25.2, 23.1, 24.5],
    'Humidity': [45.2, 45.2, 46.1, 44.8, 44.8, 47.2, 48.5, 48.5, 45.9, 47.8]
}

df_timeseries = pd.DataFrame(time_data)
print("Time series data with potential duplicates:")
print(df_timeseries)

# Find exact duplicates in sensor readings
sensor_duplicates = df_timeseries.duplicated(subset=['Sensor_ID', 'Temperature', 'Humidity'])
print(f"\nExact sensor reading duplicates: {sensor_duplicates.sum()}")

# Keep latest reading for each sensor when values are identical
df_ts_dedup = df_timeseries.drop_duplicates(
    subset=['Sensor_ID', 'Temperature', 'Humidity'], 
    keep='last'
)
print("After removing duplicate sensor readings (keeping latest):")
print(df_ts_dedup)

# 6.2 HIERARCHICAL DUPLICATES
print("\n\n6.2 HANDLING HIERARCHICAL DUPLICATES")
print("Method: Deal with duplicates at different levels of hierarchy")

# Create hierarchical data
hierarchy_data = {
    'Country': ['USA', 'USA', 'USA', 'Canada', 'Canada', 'USA', 'Mexico', 'Mexico'],
    'State': ['CA', 'CA', 'NY', 'ON', 'BC', 'CA', 'DF', 'DF'],
    'City': ['Los Angeles', 'San Francisco', 'New York', 'Toronto', 'Vancouver', 'Los Angeles', 'Mexico City', 'Mexico City'],
    'Population': [4000000, 875000, 8400000, 2930000, 675000, 4000000, 9200000, 9200000],
    'Year': [2023, 2023, 2023, 2023, 2023, 2024, 2023, 2024]
}

df_hierarchy = pd.DataFrame(hierarchy_data)
print("Hierarchical data:")
print(df_hierarchy)

# Remove duplicates at city level (keeping most recent year)
df_hierarchy_sorted = df_hierarchy.sort_values(['Country', 'State', 'City', 'Year'])
df_city_unique = df_hierarchy_sorted.drop_duplicates(
    subset=['Country', 'State', 'City'], 
    keep='last'
)
print("\nAfter removing city-level duplicates (keeping latest year):")
print(df_city_unique)

# =============================================================================
# 7. BEST PRACTICES AND RECOMMENDATIONS
# =============================================================================

print("\n\n7. BEST PRACTICES AND RECOMMENDATIONS")
print("-" * 50)

recommendations = """
DUPLICATE HANDLING BEST PRACTICES:

1. UNDERSTAND YOUR DUPLICATES:
   - Are they exact duplicates or near-duplicates?
   - What caused the duplicates (data entry errors, system issues)?
   - Which fields should be considered for duplicate detection?

2. DETECTION STRATEGIES:
   
   For EXACT duplicates:
   - Use duplicated() for complete row duplicates
   - Use subset parameter for key column duplicates
   - Consider keep='first', 'last', or False based on needs
   
   For FUZZY duplicates:
   - Use string similarity algorithms (Levenshtein, Jaccard)
   - Consider phonetic matching for names
   - Use domain-specific rules (email normalization)

3. REMOVAL STRATEGIES:
   
   KEEP FIRST: When order matters (chronological data)
   KEEP LAST: When latest information is most accurate
   KEEP NONE: When all duplicates are problematic
   CONDITIONAL: When business rules determine which to keep
   AGGREGATE: When you want to combine information

4. VALIDATION CHECKLIST:
   ✓ Verify expected number of records removed
   ✓ Check that key identifiers are now unique
   ✓ Analyze impact on data distributions
   ✓ Validate business logic is preserved
   ✓ Document deduplication decisions

5. COMMON SCENARIOS:

   CUSTOMER DATA:
   - Deduplicate by customer ID or email
   - Keep most recent or most complete record
   - Consider aggregating transaction history
   
   TIME SERIES:
   - Remove exact timestamp duplicates
   - Keep latest reading for sensor data
   - Consider time windows for near-duplicates
   
   PRODUCT CATALOGS:
   - Deduplicate by SKU or product code
   - Merge product descriptions and attributes
   - Handle variant products carefully

6. THINGS TO AVOID:
   ✗ Don't remove duplicates without understanding why they exist
   ✗ Don't use drop_duplicates() on entire DataFrame without analysis
   ✗ Don't ignore the business impact of removing records
   ✗ Don't forget to validate results after deduplication
   ✗ Don't assume all duplicates are errors

7. PERFORMANCE TIPS:
   - Sort data before deduplication for consistent results
   - Use subset parameter to focus on relevant columns
   - Consider chunking for very large datasets
   - Index key columns for faster duplicate detection

8. DOCUMENTATION:
   - Record deduplication criteria and business rules
   - Log number of duplicates found and removed
   - Keep examples of duplicate patterns for future reference
   - Document any manual review processes
"""

print(recommendations)

# =============================================================================
# 8. SUMMARY COMPARISON
# =============================================================================

print("\n\n8. SUMMARY COMPARISON OF METHODS")
print("-" * 50)

# Create summary comparison
methods_comparison = {
    'Method': [
        'Complete Row Duplicates',
        'Key Column Duplicates (first)',
        'Key Column Duplicates (last)', 
        'Conditional (max value)',
        'Aggregation',
        'Remove All Duplicates'
    ],
    'Original_Records': [len(df_original)] * 6,
    'Final_Records': [
        len(df_original.drop_duplicates()),
        len(df_unique_customers_first),
        len(df_unique_customers_last),
        len(df_max_purchase),
        len(df_aggregated),
        len(df_remove_all_dups)
    ],
    'Records_Removed': [
        len(df_original) - len(df_original.drop_duplicates()),
        len(df_original) - len(df_unique_customers_first),
        len(df_original) - len(df_unique_customers_last),
        len(df_original) - len(df_max_purchase),
        len(df_original) - len(df_aggregated),
        len(df_original) - len(df_remove_all_dups)
    ],
    'Use_Case': [
        'Identical rows',
        'Keep oldest record',
        'Keep newest record',
        'Keep best record',
        'Combine records',
        'Remove all duplicates'
    ]
}

comparison_df = pd.DataFrame(methods_comparison)
print("Comparison of duplicate handling methods:")
print(comparison_df)

print("\n" + "=" * 60)
print("END OF DUPLICATE HANDLING DEMONSTRATION")
print("=" * 60)

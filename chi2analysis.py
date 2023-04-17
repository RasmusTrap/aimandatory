import pandas as pd
from scipy.stats import chi2_contingency

# Load the heart disease dataset
heart_df = pd.read_csv('Heart_disease_statlog.csv')

# Group age into 4 age groups
age_groups = pd.cut(heart_df['age'], bins=[0, 40, 50, 60, 100], labels=['<40', '40-50', '50-60', '60+'])
heart_df['age_group'] = age_groups

# Create a contingency table
contingency_table = pd.crosstab(heart_df['age_group'], heart_df['target'])

# Perform the chi-square test
chi2, p_value, dof, expected = chi2_contingency(contingency_table)

# Print the results
print("Expected frequencies:")
print(pd.DataFrame(expected, index=contingency_table.index, columns=contingency_table.columns))
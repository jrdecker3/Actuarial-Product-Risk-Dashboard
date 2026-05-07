import pandas as pd
import numpy as np

print("--- Phase 3: Multi-Company Data Engineering ---")

# 1. Load the raw data
file_path = 'data/prodliab_pos_98-07.csv'
raw_data = pd.read_csv(file_path)

print(f"Loaded {len(raw_data)} raw rows across {raw_data['GRNAME'].nunique()} companies.")

# 2. Sort the data globally

sorted_data = raw_data.sort_values(['GRNAME', 'AccidentYear', 'DevelopmentLag'])

# 3. Calculate Incremental Paid Loss safely across ALL companies
sorted_data['IncrementalPaid'] = sorted_data.groupby(['GRNAME', 'AccidentYear'])['CumPaidLoss'].diff()
sorted_data['IncrementalPaid'] = sorted_data['IncrementalPaid'].fillna(sorted_data['CumPaidLoss'])

# 4. Filter out zeros and negatives (required for distribution fitting)
clean_data = sorted_data[sorted_data['IncrementalPaid'] > 0].copy()

# 5. Check Viability

company_counts = clean_data.groupby('GRNAME').size()
valid_companies = company_counts[company_counts >= 15].index
final_data = clean_data[clean_data['GRNAME'].isin(valid_companies)]

print(f"\nFiltered down to {final_data['GRNAME'].nunique()} viable companies (dropped those with insufficient data).")
print(f"Total rows ready for modeling: {len(final_data)}")

# 6. Save the master cleaned dataset
output_path = 'data/cleaned_cas_data.csv'
final_data.to_csv(output_path, index=False)

print(f"\n SUCCESS! Master dataset saved to '{output_path}'.")
print("Data pipeline complete. Ready for the dashboard.")
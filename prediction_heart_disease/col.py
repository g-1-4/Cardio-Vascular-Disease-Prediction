import pandas as pd
cleaned_file_path = 'Cleaned_data.csv'
data = pd.read_csv(cleaned_file_path)
data['age_years'] = data['age'] / 365
data['MAP'] = data['ap_lo'] +  ( data['ap_hi']-data['ap_lo']) / 3
updated_file_path = 'Updated_data.csv'
data.to_csv(updated_file_path, index=False)
print(data.head())

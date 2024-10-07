import pandas as pd
clustered_file_path = 'Updated_data.csv'
data = pd.read_csv(clustered_file_path)
columns_to_remove = [
    'ap_hi', 'ap_lo', 
    'calculated_bmi', 'age'
]
data = data.drop(columns=columns_to_remove)
final_file_path = 'last_data.csv'
data.to_csv(final_file_path, index=False)
print(data.head())

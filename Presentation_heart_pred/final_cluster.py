import pandas as pd
clustered_file_path = 'Clustered_2.csv'
data = pd.read_csv(clustered_file_path)
columns_to_remove = [
    'height','height(m)','weight','bmi','age_years','MAP'
]
data = data.drop(columns=columns_to_remove)
data.to_csv('final_clustered_data.csv', index=False)
print(data.head())
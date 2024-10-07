import pandas as pd
from kmodes.kmodes import KModes
updated_file_path = 'Cleaned_data.csv'
data = pd.read_csv(updated_file_path)
data['age_bins'] = pd.cut(data['age_years'], bins=[0, 20, 30, 40, 50, 60, 70, 80, 90, 100], labels=False)
data['height_bins'] = pd.cut(data['height'], bins=5, labels=False)
data['weight_bins'] = pd.cut(data['weight'], bins=5, labels=False)
data['bmi_bins'] = pd.cut(data['bmi'], bins=5, labels=False)
data['MAP_bins'] = pd.cut(data['MAP'], bins=5, labels=False)
categorical_columns = ['age_bins', 'gender', 'height_bins', 'weight_bins', 'cholesterol', 'gluc', 'smoke', 'alco', 'active', 'cardio']
categorical_data = data[categorical_columns].copy()
n_clusters = 5  
km = KModes(n_clusters=n_clusters, init='Huang', n_init=5, verbose=1)
clusters = km.fit_predict(categorical_data)
data['cluster'] = clusters
clustered_file_path = 'Clustered_2.csv'
data.to_csv(clustered_file_path, index=False)
print(data.describe())

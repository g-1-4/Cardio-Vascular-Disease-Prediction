import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import zscore
df = pd.read_csv("updated datset.csv")
def detect_outliers_zscore(data, threshold=3):
    z_scores = np.abs(zscore(data))
    return np.where(z_scores > threshold)
outlier_indices = {}
for column in ['height', 'weight', 'bmi', 'ap_hi', 'ap_lo']:
    outliers = detect_outliers_zscore(df[column])
    outlier_indices[column] = outliers[0]
    print(f" {column}: {outliers[0]}")
df['calculated_bmi'] = df['weight'] / ((df['height'] / 100) ** 2)
inconsistent_bmi = df[np.abs(df['bmi'] - df['calculated_bmi']) > 1] 
print(inconsistent_bmi.index)
inconsistent_bp = df[df['ap_hi'] <= df['ap_lo']]
print(inconsistent_bp.index)
indices_to_remove = set(outlier_indices['height']).union(
    set(outlier_indices['weight']),
    set(outlier_indices['bmi']),
    set(outlier_indices['ap_hi']),
    set(outlier_indices['ap_lo']),
    set(inconsistent_bmi.index),
    set(inconsistent_bp.index)
)
df_cleaned = df.drop(indices_to_remove)
print(f"Cleaned data shape: {df_cleaned.shape}")
print(df_cleaned.describe())
df_cleaned.to_csv('cleaned_dataset.csv', index=False)
import pandas as pd
import numpy as np
file_path = 'last_data.csv'
data = pd.read_csv(file_path)
data_cleaned = data.dropna()
def cap_outliers(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    df[column] = np.where(df[column] < lower_bound, lower_bound, df[column])
    df[column] = np.where(df[column] > upper_bound, upper_bound, df[column])
    return df
for col in data_cleaned.select_dtypes(include=[np.number]).columns:
    data_cleaned = cap_outliers(data_cleaned, col)
num_rows_cleaned = data_cleaned.shape[0]
print(f'Number of rows after cleaning: {num_rows_cleaned}')
cleaned_file_path = 'cleaned_data.csv'
data_cleaned.to_csv(cleaned_file_path, index=False)
print(f'Cleaned data saved to {cleaned_file_path}')

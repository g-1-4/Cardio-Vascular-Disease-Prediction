import pandas as pd
data = pd.read_csv('cardio_train.csv', delimiter=';')
def remove_outliers_iqr(df, columns):
    for col in columns:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]
    return df

# Columns to check for outliers
columns_to_check = ['height', 'weight', 'ap_hi', 'ap_lo']

# Removing outliers
cleaned_data = remove_outliers_iqr(data, columns_to_check)

cleaned_data.to_csv('cleaned_2.csv',index=False)

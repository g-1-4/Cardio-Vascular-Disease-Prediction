import pandas as pd
df = pd.read_csv('Preprocessed_dataset.csv')
df_cleaned = df[df['ap_hi'] <= 350]
df_cleaned_min = df_cleaned[df_cleaned['ap_lo']>=35]
max=df_cleaned['bmi'].max()
min=df_cleaned_min['ap_lo'].min()
min=df_cleaned_min['height'].min()
print(max)
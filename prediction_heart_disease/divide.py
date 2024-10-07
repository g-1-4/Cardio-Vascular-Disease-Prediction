import pandas as pd
df = pd.read_csv('Final_data.csv')
male_df = df[df['gender'] == 1]
female_df = df[df['gender'] == 2]
male_df.to_csv('male_dataset.csv', index=False)
female_df.to_csv('female_dataset.csv', index=False)
print("Datasets have been split and saved as 'male_dataset.csv' and 'female_dataset.csv'.")

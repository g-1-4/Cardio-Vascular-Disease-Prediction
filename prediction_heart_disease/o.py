import pandas as pd
from scipy.stats import zscore

# Load the dataset
file_path = 'cardio_train.csv'
data = pd.read_csv(file_path)

# Define function to remove outliers using Z-score
def remove_outliers_zscore(df, threshold=4):
    z_scores = zscore(df.select_dtypes(include=[float, int]), nan_policy='omit')
    abs_z_scores = abs(z_scores)
    filtered_entries = (abs_z_scores < threshold).all(axis=1)
    return df[filtered_entries]

# Initial filtering conditions
conditions = (
    (data['height'] > 120) & (data['height'] < 220) &  # Reasonable height range
    (data['weight'] > 30) & (data['weight'] < 200) &  # Reasonable weight range
    (data['ap_hi'] > 50) & (data['ap_hi'] < 250) &    # Reasonable systolic BP
    (data['ap_lo'] > 30) & (data['ap_lo'] < 150)      # Reasonable diastolic BP
)

# Apply the conditions to filter the dataset
filtered_data = data[conditions]

# Remove outliers with Z-score threshold
filtered_data = remove_outliers_zscore(filtered_data, threshold=3)

# Check if we have enough data points, if not, relax conditions
if filtered_data.shape[0] < 61157:
    relaxed_conditions = (
        (data['height'] > 130) & (data['height'] < 210) &  # Slightly expanded height range
        (data['weight'] > 40) & (data['weight'] < 180) &  # Slightly expanded weight range
        (data['ap_hi'] > 80) & (data['ap_hi'] < 200) &    # Adjusted systolic BP range
        (data['ap_lo'] > 50) & (data['ap_lo'] < 120)      # Adjusted diastolic BP range
    )

    relaxed_filtered_data = data[relaxed_conditions]
    relaxed_filtered_data = remove_outliers_zscore(relaxed_filtered_data, threshold=4)

    # Check again and sample
    if relaxed_filtered_data.shape[0] >= 61157:
        final_data = relaxed_filtered_data.sample(n=61157, random_state=42)
    else:
        print("Not enough data points after relaxing the conditions. Please adjust further.")
else:
    final_data = filtered_data.sample(n=61157, random_state=42)

# Save the final dataset to a CSV file
final_data.to_csv('finali_dataset.csv', index=False)

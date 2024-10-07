import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Load the male dataset
male_df = pd.read_csv('male_dataset.csv')

# Select features and target variable
X_male = male_df.drop(columns=['cardio','smoke','alco', 'id','gender','height_bins', 'weight_bins'])
y_male = male_df['cardio']

# Split the data into training and testing sets
X_train_male, X_test_male, y_train_male, y_test_male = train_test_split(X_male, y_male, test_size=0.2, random_state=42)

# Standardize the features
scaler = StandardScaler()
X_train_male_scaled = scaler.fit_transform(X_train_male)
X_test_male_scaled = scaler.transform(X_test_male)

# Initialize and train the model
model_male = LogisticRegression()
model_male.fit(X_train_male_scaled, y_train_male)

# Predict and evaluate
y_pred_male = model_male.predict(X_test_male_scaled)
accuracy_male = accuracy_score(y_test_male, y_pred_male)

print(f'Male Dataset Logistic Regression Accuracy: {accuracy_male:.4f}')

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, classification_report, auc, roc_curve
import numpy as np
import matplotlib.pyplot as plt


data = pd.read_csv('final_clustered_data.csv')
X = data.drop(columns=['cardio', 'id'])
y = data['cardio']
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42, stratify=y)
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)
y_pred = lr_model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"Linear Regression Mean Squared Error: {mse:.5f}")
print(f"Linear Regression R^2 Score: {r2:.5f}")
y_pred_binary = np.where(y_pred > 0.5, 1, 0)
accuracy = accuracy_score(y_test, y_pred_binary) * 100
print(f"Linear Regression Accuracy: {accuracy:.5f}")
print("\nLinear Regression Classification Report:")
print(classification_report(y_test, y_pred_binary))

y_prob = lr_model.predict_proba(X_test)[:, 1]
fpr, tpr, thresholds = roc_curve(y_test, y_prob)
roc_auc = auc(fpr, tpr)
print(f"Random Forest AUC: {roc_auc:.5f}")
plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.5f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC)')
plt.legend(loc="lower right")
plt.show()
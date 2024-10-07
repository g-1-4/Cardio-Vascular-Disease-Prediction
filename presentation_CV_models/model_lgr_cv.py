import pandas as pd
data=pd.read_csv('final_clustered_data.csv')
X = data.drop(columns=['cardio', 'id'])
y = data['cardio']
from sklearn.preprocessing import StandardScaler
scaler=StandardScaler()
from sklearn.model_selection import train_test_split, GridSearchCV
X_scaled = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.4, random_state=42, stratify=y)
from sklearn.metrics import accuracy_score, classification_report
from sklearn.linear_model import LogisticRegression
param_grid_lr = {
    'C': [0.1, 1, 10],
    'penalty': ['l2'],
    'solver': ['lbfgs', 'liblinear']
}
lr = LogisticRegression(random_state=42, max_iter=10000)
grid_search_lr = GridSearchCV(estimator=lr, param_grid=param_grid_lr, cv=3, n_jobs=-1, verbose=2)
grid_search_lr.fit(X_train, y_train)
print("Best parameters for Logistic Regression:", grid_search_lr.best_params_)
y_pred_lr = grid_search_lr.best_estimator_.predict(X_test)
accuracy_lr = accuracy_score(y_test, y_pred_lr)
print(f"Logistic Regression Accuracy: {accuracy_lr:.5f}")
print("\nLogistic Regression Classification Report:")
print(classification_report(y_test, y_pred_lr))

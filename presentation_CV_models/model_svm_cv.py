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
from sklearn.svm import SVC
param_grid_svc = {
    'C': [0.1, 1, 10],
    'kernel': ['linear', 'rbf'],
    'gamma': ['scale', 'auto']
}
svc = SVC(random_state=42)
grid_search_svc = GridSearchCV(estimator=svc, param_grid=param_grid_svc, cv=3, n_jobs=-1, verbose=2)
grid_search_svc.fit(X_train, y_train)
print("Best parameters for SVM:", grid_search_svc.best_params_)
y_pred_svc = grid_search_svc.best_estimator_.predict(X_test)
accuracy_svc = accuracy_score(y_test, y_pred_svc)
print(f"SVM Accuracy: {accuracy_svc:.5f}")
print("\nSVM Classification Report:")
print(classification_report(y_test, y_pred_svc))

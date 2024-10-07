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
from sklearn.neighbors import KNeighborsClassifier
param_grid_knn = {
    'n_neighbors': [3, 5, 7, 9],
    'weights': ['uniform', 'distance'],
    'metric': ['euclidean', 'manhattan']
}
knn = KNeighborsClassifier()
grid_search_knn = GridSearchCV(estimator=knn, param_grid=param_grid_knn, cv=3, n_jobs=-1, verbose=2)
grid_search_knn.fit(X_train, y_train)
print("Best parameters for K Nearest Neighbors:", grid_search_knn.best_params_)
y_pred_knn = grid_search_knn.best_estimator_.predict(X_test)
accuracy_knn = accuracy_score(y_test, y_pred_knn)
print(f"K Nearest Neighbors Accuracy: {accuracy_knn:.5f}")
print("\nK Nearest Neighbors Classification Report:")
print(classification_report(y_test, y_pred_knn))

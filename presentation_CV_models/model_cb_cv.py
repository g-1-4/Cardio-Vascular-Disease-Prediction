import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
from catboost import CatBoostClassifier
data = pd.read_csv('final_clustered_data.csv')
X = data.drop(columns=['cardio', 'id'])  
y = data['cardio']
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42, stratify=y)
model = CatBoostClassifier(verbose=0)
param_grid_cb = {
    'iterations': [100, 200],
    'learning_rate': [0.1,0.5],
    'depth': [4, 6],
    'l2_leaf_reg': [1, 3, 5, 7, 9]
}
grid_search = GridSearchCV(estimator=model, param_grid=param_grid_cb, cv=3, n_jobs=-1, scoring='accuracy')
grid_search.fit(X_train, y_train)
best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred) * 100
print(f"CatBoost Accuracy: {accuracy:.5f}")
print("\nCatBoost Classification Report:")
print(classification_report(y_test, y_pred))
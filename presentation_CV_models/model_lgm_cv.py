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
import lightgbm as lgb
param_grid_lgbm = {
    'n_estimators': [50, 100],
    'max_depth': [10, 20, -1],
    'num_leaves': [31, 40],
    'min_data_in_leaf': [20, 50]
}
lgbm = lgb.LGBMClassifier(random_state=42)
grid_search_lgbm = GridSearchCV(estimator=lgbm, param_grid=param_grid_lgbm, cv=3, n_jobs=-1, verbose=2)
grid_search_lgbm.fit(X_train, y_train)
print("Best parameters for LightGBM:", grid_search_lgbm.best_params_)
y_pred_lgbm = grid_search_lgbm.best_estimator_.predict(X_test)
accuracy_lgbm = accuracy_score(y_test, y_pred_lgbm)
print(f"LightGBM Accuracy: {accuracy_lgbm:.5f}")
print("\nLightGBM Classification Report:")
print(classification_report(y_test, y_pred_lgbm))
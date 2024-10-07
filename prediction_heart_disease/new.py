import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report
file_path = 'Clustered_2.csv'
data = pd.read_csv(file_path)
X = data.drop(columns=['id', 'cardio'])
y = data['cardio']
categorical_features = ['gender', 'cholesterol', 'gluc', 'smoke', 'alco', 'active', 'age_bins', 'height_bins', 'weight_bins', 'bmi_bins', 'MAP_bins', 'cluster']
numerical_features = []
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_features),
        ('cat', OneHotEncoder(), categorical_features)
    ]
    )
def evaluate_model(model, X_train, X_test, y_train, y_test):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    return accuracy, report
models = {
    'RandomForest': {
        'model': RandomForestClassifier(random_state=42),
        'params': {
            'clf__n_estimators': [100, 200],
            'clf__max_depth': [10, 20],
            'clf__min_samples_split': [2, 5],
            'clf__min_samples_leaf': [1, 2]
        }
    },
    'XGBoost': {
        'model': XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42),
        'params': {
            'clf__n_estimators': [100, 200],
            'clf__max_depth': [10, 20],
            'clf__learning_rate': [0.01, 0.1],
            'clf__subsample': [0.8, 1.0]
        }
    },
    'GradientBoosting': {
        'model': GradientBoostingClassifier(random_state=42),
        'params': {
            'clf__n_estimators': [100, 200],
            'clf__max_depth': [10, 20],
            'clf__learning_rate': [0.01, 0.1],
            'clf__subsample': [0.8, 1.0]
        }
    }
}
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
best_model = None
best_accuracy = 0
best_report = ""


for model_name, model_info in models.items():
    print(f"Training {model_name}...")
    pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('clf', model_info['model'])])
    grid_search = GridSearchCV(pipeline, model_info['params'], cv=3, n_jobs=-1, verbose=2)
    grid_search.fit(X_train, y_train)
    accuracy, report = evaluate_model(grid_search.best_estimator_, X_train, X_test, y_train, y_test)
    print(f"{model_name} Accuracy: ", accuracy)
    print(f"{model_name} Classification Report:\n", report)
    if accuracy > best_accuracy:
        best_model = grid_search.best_estimator_
        best_accuracy = accuracy
        best_report = report
print("Best Model Accuracy: ", best_accuracy)
print("Best Model Classification Report:\n", best_report)

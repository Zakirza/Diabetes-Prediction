from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV

def train_base_models(X_train, y_train):
    models = {
        "RandomForest": RandomForestClassifier(),
        "SVM": SVC(probability=True),
        "Logistic": LogisticRegression(max_iter=2000),
        "KNN": KNeighborsClassifier(),
        "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='logloss')
    }

    tuned_models = {}
    for name, model in models.items():
        print(f" Tuning {name}...")
        param_grid = {
            "RandomForest": {"n_estimators": [100, 200]},
            "SVM": {"C": [0.1, 1, 10]},
            "KNN": {"n_neighbors": [3, 5, 7]},
            "Logistic": {"C": [0.1, 1, 10]},
            "XGBoost": {"learning_rate": [0.05, 0.1], "max_depth": [3, 5]}
        }.get(name, {})

        grid = GridSearchCV(model, param_grid, cv=3, scoring="accuracy", n_jobs=-1)
        grid.fit(X_train, y_train)
        tuned_models[name] = grid.best_estimator_
        print(f" {name} tuned. Best score: {grid.best_score_:.3f}")

    return tuned_models

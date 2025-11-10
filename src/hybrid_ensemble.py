from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression

def build_hybrid_ensemble(base_models, X_train, y_train):
    print(" Building hybrid ensemble using StackingClassifier...")

    estimators = [(name, model) for name, model in base_models.items()]
    meta_model = StackingClassifier(
        estimators=estimators,
        final_estimator=LogisticRegression(),
        passthrough=True
    )

    meta_model.fit(X_train, y_train)
    print(" Hybrid ensemble trained successfully.")
    return meta_model

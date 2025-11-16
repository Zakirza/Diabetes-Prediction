from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression

def build_hybrid_ensemble(base_models, X_train, y_train):
    print("\n🧩 Building hybrid ensemble...")

    ml_models = [(name, model) for name, model in base_models.items()
                 if name not in ["ANN", "CNN", "ANN_Tuned", "CNN_Tuned"]]

    stacker = StackingClassifier(
        estimators=ml_models,
        final_estimator=LogisticRegression(max_iter=3000),
        passthrough=True
    )

    stacker.fit(X_train, y_train)
    print("✅ Hybrid ensemble trained successfully.")
    return stacker

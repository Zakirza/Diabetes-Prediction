from sklearn.ensemble import StackingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression

def build_hybrid_ensemble(base_models, X_train, y_train):
    print("\n🧩 Building hybrid ensemble...")

    # Base ML models only (exclude deep ones)
    estimators = [(n, m) for n, m in base_models.items() if n not in ["ANN", "CNN", "ANN_Tuned", "CNN_Tuned"]]

    # Stacking model
    stacker = StackingClassifier(
        estimators=estimators,
        final_estimator=LogisticRegression(max_iter=2000),
        passthrough=True
    )
    stacker.fit(X_train, y_train)

    # Weighted Voting ensemble
    voter = VotingClassifier(
        estimators=estimators,
        voting='soft',
        weights=[2, 2, 1, 1, 3]  # give higher weight to strong models
    )
    voter.fit(X_train, y_train)

    print(" Hybrid ensemble trained successfully.")
    return stacker

import numpy as np
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score, confusion_matrix
import matplotlib.pyplot as plt

def evaluate_models(base_models, hybrid_model, X_test, y_test):
    print("\n Evaluating models...")

    # Evaluate base models individually
    for name, model in base_models.items():
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        print(f"{name} Accuracy: {acc:.3f}")

    # Evaluate hybrid (StackingClassifier)
    hybrid_pred = hybrid_model.predict(X_test)

    print(f"\nHybrid Ensemble Accuracy: {accuracy_score(y_test, hybrid_pred):.3f}")
    print(f"ROC-AUC: {roc_auc_score(y_test, hybrid_pred):.3f}")
    print("\nClassification Report:\n", classification_report(y_test, hybrid_pred))

    # Confusion Matrix Visualization
    cm = confusion_matrix(y_test, hybrid_pred)
    plt.imshow(cm, cmap="Blues")
    plt.title("Confusion Matrix")
    plt.colorbar()
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.savefig("results/confusion_matrix.png")
    plt.close()

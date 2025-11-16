import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score, RocCurveDisplay

def evaluate_models(base_models, hybrid_model, X_test, y_test):
    print("\n🔍 Evaluating models...")
    X_test_cnn = np.expand_dims(X_test, axis=2)

    for name, model in base_models.items():
        print(f"\n{name} Evaluation:")
        if name in ["CNN", "CNN_Tuned"]:
            y_pred = (model.predict(X_test_cnn) > 0.5).astype("int32")
        elif name in ["ANN", "ANN_Tuned"]:
            y_pred = (model.predict(X_test) > 0.5).astype("int32")
        else:
            y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        print(f" Accuracy: {acc:.3f}")
        print(" Classification Report:\n", classification_report(y_test, y_pred))

    print("\n🧠 Hybrid Ensemble Evaluation:")
    hybrid_pred = hybrid_model.predict(X_test)
    print(f" Accuracy: {accuracy_score(y_test, hybrid_pred):.3f}")
    print(f" ROC-AUC: {roc_auc_score(y_test, hybrid_pred):.3f}")

    cm = confusion_matrix(y_test, hybrid_pred)
    plt.imshow(cm, cmap="Blues")
    plt.title("Confusion Matrix - Hybrid Ensemble")
    plt.colorbar()
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.savefig("results/confusion_matrix.png")
    plt.close()

    RocCurveDisplay.from_estimator(hybrid_model, X_test, y_test)
    plt.title("ROC Curve - Hybrid Ensemble")
    plt.show()

    print("\n📊 Confusion Matrix saved to results/confusion_matrix.png")


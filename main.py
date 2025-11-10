from src.data_preprocessing import load_and_preprocess_data
from src.model_training import train_base_models
from src.hybrid_ensemble import build_hybrid_ensemble
from src.evaluation import evaluate_models

def main():
    print(" Starting Hybrid Ensemble for Diabetes Prediction")

    # Step 1: Data Preparation--
    X_train, X_test, y_train, y_test = load_and_preprocess_data("data/diabetes.csv")

    # Step 2: Train base models--
    base_models = train_base_models(X_train, y_train)

    # Step 3: Hybrid Ensemble--
    hybrid_model = build_hybrid_ensemble(base_models, X_train, y_train)

    # Step 4: Evaluation--
    evaluate_models(base_models, hybrid_model, X_test, y_test)

    print(" All steps completed successfully!")

if __name__ == "__main__":
    main()

from src.data_preprocessing import load_and_preprocess_data
from src.model_training import train_base_models
from src.hybrid_ensemble import build_hybrid_ensemble
from src.evaluation import evaluate_models
from src.fine_tuning import tune_ann_model, tune_cnn_model

def main():
    print("🚀 Starting Hybrid Ensemble Diabetes Prediction")

    # Load data
    (X_train, X_test, y_train, y_test), scaler = load_and_preprocess_data("data/Diabetes.csv")

    # Train ML + ANN + CNN
    base_models = train_base_models(X_train, y_train)

    # Fine-tuning (RUN ONLY ONCE — SAVES BEST MODELS)
    print("\n🔧 Fine-tuning ANN...")
    best_ann = tune_ann_model(X_train, y_train)

    print("\n🔧 Fine-tuning CNN...")
    best_cnn = tune_cnn_model(X_train, y_train)

    base_models["ANN_Tuned"] = best_ann
    base_models["CNN_Tuned"] = best_cnn

    # Build hybrid ensemble (Stacking)
    hybrid_model = build_hybrid_ensemble(base_models, X_train, y_train)

    # Evaluate everything
    evaluate_models(base_models, hybrid_model, X_test, y_test)

    print("\n🎉 All steps completed successfully!")

if __name__ == "__main__":
    main()

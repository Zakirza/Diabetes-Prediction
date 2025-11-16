from src.data_preprocessing import load_and_preprocess_data
from src.model_training import train_base_models
from src.hybrid_ensemble import build_hybrid_ensemble
from src.evaluation import evaluate_models
from src.fine_tuning import tune_ann_model, tune_cnn_model

def main():
    print("🚀 Starting Advanced Hybrid Ensemble for Diabetes Prediction")

    # Step 1: Load & preprocess data
    X_train, X_test, y_train, y_test = load_and_preprocess_data("data/diabetes.csv")

    # Step 2: Train base ML + Deep models
    base_models = train_base_models(X_train, y_train)

    # Step 3: Fine-tune ANN and CNN using Keras Tuner
    print("\n🔧 Fine-tuning ANN model...")
    best_ann = tune_ann_model(X_train, y_train)

    print("\n🔧 Fine-tuning CNN model...")
    best_cnn = tune_cnn_model(X_train, y_train)

    base_models["ANN_Tuned"] = best_ann
    base_models["CNN_Tuned"] = best_cnn

    # Step 4: Build Hybrid Ensemble
    hybrid_model = build_hybrid_ensemble(base_models, X_train, y_train)

    # Step 5: Evaluate
    evaluate_models(base_models, hybrid_model, X_test, y_test)

    print("\n All steps completed successfully!")

if __name__ == "__main__":
    main()


# ⭐ **README.md — Diabetes Prediction using Hybrid Ensemble (ML + ANN + CNN)**

# 📌 Diabetes Prediction – Hybrid Machine Learning + Deep Learning Ensemble

A complete end-to-end **medical risk prediction system** using:

* Classical ML Models
* Artificial Neural Network (ANN)
* Convolutional Neural Network (CNN)
* Keras Tuner for hyperparameter optimization
* Hybrid Stacking Ensemble

This project predicts whether a person is diabetic based on medical features (PIMA Indians Diabetes Dataset).
The pipeline includes **EDA → Preprocessing → Feature Engineering → ML/DL Models → Fine-Tuning → Hybrid Ensemble → Evaluation**.

---

# 🚀 **Project Features**

### 1. Full end-to-end ML/DL pipeline

### 2. Advanced preprocessing & feature engineering

### 3. ANN + 1D-CNN models with BatchNorm, Dropout

### 4. Hyperparameter tuning using Keras Tuner

### 5. Hybrid Stacking Ensemble (best ML models)

### 6. SMOTE balancing for medical datasets

### 7. Confusion Matrix + ROC Curve visualizations

### 8. Modular, scalable, production-ready architecture

---

# 🧠 **Project Architecture**

```
data/
    Diabetes.csv

src/
    data_preprocessing.py
    model_training.py
    fine_tuning.py
    hybrid_ensemble.py
    evaluation.py

results/
    confusion_matrix.png
    ann_tuning/
    cnn_tuning/

main.py
README.md
```

---

# 📊 **1. Exploratory Data Analysis (EDA)**

Performed a full EDA including:

* ✔ Missing values inspection
* ✔ Handling invalid zeros (Glucose, Insulin, BP, BMI…)
* ✔ Distribution plots
* ✔ Boxplots for outliers
* ✔ Correlation heatmap
* ✔ Outcome class imbalance check
* ✔ Feature relationships
* ✔ Summary statistics

**Key EDA Insights:**

* Glucose shows the strongest correlation with diabetes.
* Insulin and SkinThickness contain many missing/zero values.
* Age & BMI increase diabetes probability.
* Dataset is imbalanced → requires SMOTE.

---

# 🧹 **2. Data Preprocessing**

### ✔ Replace zero values with NaN

### ✔ Median imputation

### ✔ Feature scaling (StandardScaler)

### ✔ SMOTE oversampling

### ✔ Feature engineering:

```
Glucose_BMI = Glucose * BMI
Age_BP = Age * BloodPressure
Insulin_sqrt = sqrt(Insulin)
```

These engineered features significantly improved ML & ANN performance.

---

# 🤖 **3. Machine Learning Models**

Trained 5 ML models:

* Random Forest
* Support Vector Machine (SVM)
* Logistic Regression
* K-Nearest Neighbors (KNN)
* XGBoost

All models were trained using:

* Scaled data
* Balanced data
* Cross-validation
* GridSearch-like tuning

---

# 🧬 **4. Deep Learning Models**

### 🟦 **4.1 ANN Model**

Architecture:

* Dense(128, relu) + BatchNorm + Dropout
* Dense(64, relu) + BatchNorm + Dropout
* Dense(16, relu)
* Dense(1, sigmoid)

### 🟨 **4.2 CNN Model (1D CNN)**

* Conv1D(128) + BatchNorm + Dropout
* Conv1D(64)
* Flatten
* Dense(32)
* Output: Dense(1, sigmoid)

### 🟣 **4.3 Model Regularization**

* **Batch Normalization**
* **Dropout (0.3–0.5)**
* **EarlyStopping(patience=10)**

---

# 🔧 **5. Hyperparameter Tuning – Keras Tuner**

Both ANN and CNN models were tuned using:

* Random search
* Learning rates
* Number of neurons
* Dropout rates
* Filters (for CNN)

Tuner optimizes: **val_accuracy**

Results are saved automatically in:

```
results/ann_tuning/
results/cnn_tuning/
```

---

# 🧩 **6. Hybrid Ensemble Model**

A **StackingClassifier** is built using the best ML models:

* Random Forest
* SVM
* Logistic Regression
* KNN
* XGBoost

Meta-learner: **Logistic Regression**

Deep learning models are *not* used in stacking (industry practice) because:

* Runtimes increase
* Stacking fails with Keras models
* DL models are used separately for comparison

---

# 📈 **7. Final Results**

| Model               | Accuracy |
| ------------------- | -------- |
| RandomForest        | 0.80     |
| SVM                 | 0.79     |
| Logistic Regression | 0.75     |
| KNN                 | 0.80     |
| XGBoost             | 0.78     |
| ANN                 | 0.80     |
| CNN                 | 0.76     |
| ANN_Tuned           | 0.76     |
| CNN_Tuned           | 0.78     |
| **Hybrid Ensemble** | **0.80** |

### ✔ Hybrid Ensemble performs as well as the best models

### ✔ ANN & CNN add nonlinear pattern learning

### ✔ Ensemble improves stability and robustness

---

# 🧮 **8. Evaluation Visuals**

### ✔ Confusion Matrix

### ✔ ROC Curve

### ✔ Classification Report

### ✔ Precision, Recall, F1-score

### ✔ AUC score

Confusion Matrix saved to:

```
results/confusion_matrix.png
```

---

# 🛠 **9. How to Run the Project**

### 🔹 Step 1: Install dependencies

```
pip install -r requirements.txt
```

### 🔹 Step 2: Activate environment

```
conda activate diabetes_env
```

### 🔹 Step 3: Run the project

```
python main.py
```

---

# 📦 **10. Technologies Used**

| Category         | Tech                        |
| ---------------- | --------------------------- |
| ML               | sklearn, XGBoost            |
| DL               | TensorFlow, Keras, SciKeras |
| Tuning           | Keras Tuner                 |
| Oversampling     | SMOTE (imblearn)            |
| Visualization    | Matplotlib                  |
| Deployment-ready | Modular code architecture   |

---

# 📚 **11. Future Improvements**

* Feature selection (mutual information / RFE)
* Optuna tuning for ML models
* SHAP explainability
* Streamlit or Flask deployment
* Model interpretability dashboard
* More feature engineering
* Cross-validation for DL models

---

# 📜 **12. License**

This project is open-source and free to use for learning, research, and academic purposes.

---

# 👨‍💻 **13. Developed By**

**Mohd Zakir**
Data Science & Machine Learning Engineering

---


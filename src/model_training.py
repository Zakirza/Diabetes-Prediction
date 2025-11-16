from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from scikeras.wrappers import KerasClassifier
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Dense, Dropout, BatchNormalization, Conv1D, Flatten
from tensorflow.keras.callbacks import EarlyStopping

# EarlyStopping
es = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

# ANN model
def create_ann_model():
    model = Sequential([
        Input(shape=(11,)),
        Dense(128, activation='relu'),
        BatchNormalization(),
        Dropout(0.4),
        Dense(64, activation='relu'),
        BatchNormalization(),
        Dropout(0.3),
        Dense(16, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    return model

# CNN model
def create_cnn_model():
    model = Sequential([
        Input(shape=(11,1)),
        Conv1D(128, kernel_size=2, activation='relu'),
        BatchNormalization(),
        Dropout(0.3),
        Conv1D(64, kernel_size=2, activation='relu'),
        Flatten(),
        Dense(32, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    return model

def train_base_models(X_train, y_train):
    print("🧠 Training ML + ANN + CNN models...")
    
    X_train_cnn = np.expand_dims(X_train, axis=2)

    models = {
        "RandomForest": RandomForestClassifier(),
        "SVM": SVC(probability=True),
        "Logistic": LogisticRegression(max_iter=3000),
        "KNN": KNeighborsClassifier(),
        "XGBoost": XGBClassifier(eval_metric="logloss"),
        "ANN": KerasClassifier(model=create_ann_model, epochs=40, batch_size=16, callbacks=[es], verbose=0),
        "CNN": KerasClassifier(model=create_cnn_model, epochs=40, batch_size=16, callbacks=[es], verbose=0)
    }

    trained = {}
    for name, model in models.items():
        print(f"\n🔹 Training {name}...")
        if name == "CNN":
            model.fit(X_train_cnn, y_train)
        else:
            model.fit(X_train, y_train)
        trained[name] = model
        print(f"✅ {name} trained successfully.")

    return trained

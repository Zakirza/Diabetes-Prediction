from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv1D, Flatten, Dropout
from scikeras.wrappers import KerasClassifier
import numpy as np

def create_ann_model():
    model = Sequential([
        Dense(64, activation='relu', input_dim=8),
        Dense(32, activation='relu'),
        Dropout(0.3),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

def create_cnn_model():
    model = Sequential([
        Conv1D(filters=64, kernel_size=2, activation='relu', input_shape=(8, 1)),
        Flatten(),
        Dense(32, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

def train_base_models(X_train, y_train):
    print(" Training ML + ANN + CNN models...")
    X_train_cnn = np.expand_dims(X_train, axis=2)

    models = {
        "RandomForest": RandomForestClassifier(),
        "SVM": SVC(probability=True),
        "Logistic": LogisticRegression(max_iter=2000),
        "KNN": KNeighborsClassifier(),
        "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='logloss'),
        "ANN": KerasClassifier(build_fn=create_ann_model, epochs=40, batch_size=16, verbose=0),
        "CNN": KerasClassifier(build_fn=create_cnn_model, epochs=40, batch_size=16, verbose=0)
    }

    tuned = {}
    for name, model in models.items():
        print(f"\n🔹 Training {name}...")
        if name == "CNN":
            model.fit(X_train_cnn, y_train)
        else:
            model.fit(X_train, y_train)
        tuned[name] = model
        print(f" {name} trained successfully.")
    return tuned

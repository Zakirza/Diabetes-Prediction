import keras_tuner as kt
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Dense, Dropout, BatchNormalization, Conv1D, Flatten

# --- ANN TUNER ---
def build_ann(hp):
    model = Sequential([
        Input(shape=(11,)),
        Dense(hp.Int('units1', 32, 128, step=32), activation='relu'),
        BatchNormalization(),
        Dropout(hp.Float("drop1", 0.2, 0.5, step=0.1)),
        Dense(hp.Int('units2', 16, 64, step=16), activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(hp.Choice("lr", [1e-2, 1e-3, 1e-4])),
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )
    return model

def tune_ann_model(X_train, y_train):
    tuner = kt.RandomSearch(
        build_ann,
        objective="val_accuracy",
        max_trials=5,
        directory="results",
        project_name="ann_tuning"
    )
    tuner.search(X_train, y_train, validation_split=0.2, epochs=30)
    print("🎯 ANN tuning complete.")
    return tuner.get_best_models(1)[0]

# --- CNN TUNER ---
def build_cnn(hp):
    model = Sequential([
        Input(shape=(11,1)),
        Conv1D(filters=hp.Choice("filters", [32, 64, 128]), kernel_size=2, activation="relu"),
        BatchNormalization(),
        Dropout(hp.Float("drop", 0.2, 0.5, step=0.1)),
        Flatten(),
        Dense(hp.Int("dense_units", 16, 64, step=16), activation="relu"),
        Dense(1, activation="sigmoid")
    ])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(hp.Choice("lr", [1e-2, 1e-3, 1e-4])),
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )
    return model

def tune_cnn_model(X_train, y_train):
    tuner = kt.RandomSearch(
        build_cnn,
        objective="val_accuracy",
        max_trials=5,
        directory="results",
        project_name="cnn_tuning"
    )
    X_train_cnn = np.expand_dims(X_train, axis=2)
    tuner.search(X_train_cnn, y_train, validation_split=0.2, epochs=30)
    print("🎯 CNN tuning complete.")
    return tuner.get_best_models(1)[0]

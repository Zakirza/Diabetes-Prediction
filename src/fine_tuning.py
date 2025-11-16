import keras_tuner as kt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Conv1D, Flatten
import numpy as np
import tensorflow as tf

# ---------------- ANN Fine-Tuning ----------------
def build_ann(hp):
    model = Sequential()
    model.add(Dense(hp.Int('units1', 32, 128, step=32), activation='relu', input_dim=8))
    model.add(Dropout(hp.Choice('dropout', [0.2, 0.3, 0.4])))
    model.add(Dense(hp.Int('units2', 16, 64, step=16), activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(
        optimizer=tf.keras.optimizers.Adam(hp.Choice('lr', [1e-2, 1e-3, 1e-4])),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    return model

def tune_ann_model(X_train, y_train):
    tuner = kt.RandomSearch(
        build_ann,
        objective='val_accuracy',
        max_trials=5,
        directory='results',
        project_name='ann_tuning'
    )
    tuner.search(X_train, y_train, epochs=30, validation_split=0.2)
    best_model = tuner.get_best_models(num_models=1)[0]
    print(" ANN tuning complete.")
    return best_model

# ---------------- CNN Fine-Tuning ----------------
def build_cnn(hp):
    model = Sequential()
    model.add(Conv1D(
        filters=hp.Choice('filters', [32, 64, 128]),
        kernel_size=hp.Choice('kernel', [2, 3]),
        activation='relu',
        input_shape=(8, 1)
    ))
    model.add(Flatten())
    model.add(Dropout(hp.Choice('dropout', [0.2, 0.3, 0.4])))
    model.add(Dense(hp.Int('dense_units', 16, 64, step=16), activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(
        optimizer=tf.keras.optimizers.Adam(hp.Choice('lr', [1e-2, 1e-3, 1e-4])),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    return model

def tune_cnn_model(X_train, y_train):
    tuner = kt.RandomSearch(
        build_cnn,
        objective='val_accuracy',
        max_trials=5,
        directory='results',
        project_name='cnn_tuning'
    )
    X_train_cnn = np.expand_dims(X_train, axis=2)
    tuner.search(X_train_cnn, y_train, epochs=30, validation_split=0.2)
    best_model = tuner.get_best_models(num_models=1)[0]
    print(" CNN tuning complete.")
    return best_model

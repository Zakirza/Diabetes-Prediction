import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE

def load_and_preprocess_data(path, test_size=0.2, random_state=42):
    print("📊 Loading and preprocessing data...")
    
    data = pd.read_csv(path)

    # Replace unrealistic zeros
    cols = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
    data[cols] = data[cols].replace(0, np.nan)
    data.fillna(data.median(), inplace=True)

    # ✨ Feature engineering
    data["Glucose_BMI"] = data["Glucose"] * data["BMI"]
    data["Age_BP"] = data["Age"] * data["BloodPressure"]
    data["Insulin_sqrt"] = np.sqrt(data["Insulin"] + 1)

    X = data.drop("Outcome", axis=1)
    y = data["Outcome"]

    # Scaling
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Balance classes
    sm = SMOTE(random_state=random_state)
    X_res, y_res = sm.fit_resample(X_scaled, y)

    return train_test_split(X_res, y_res, test_size=test_size, random_state=random_state), scaler

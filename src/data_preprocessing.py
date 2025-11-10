import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE

path = 'data\Diabetes.csv'
def load_and_preprocess_data(path):
    data = pd.read_csv(path)

    # Handle zeros as missing values (PIMA dataset)
    data[['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']] = \
        data[['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']].replace(0, pd.NA)
    data.fillna(data.median(), inplace=True)

    X = data.drop('Outcome', axis=1)
    y = data['Outcome']

    # Normalize
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Balance classes
    sm = SMOTE(random_state=42)
    X_res, y_res = sm.fit_resample(X_scaled, y)

    return train_test_split(X_res, y_res, test_size=0.2, random_state=42)

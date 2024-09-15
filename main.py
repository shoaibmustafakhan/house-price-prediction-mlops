import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import joblib


# Load dataset
data = pd.read_csv('train.csv')


def preprocess_data(df):
    # Drop columns with too many missing values or irrelevant features
    df = df.drop(['Id', 'Alley', 'PoolQC', 'Fence', 'MiscFeature'], axis=1)
    
    # Separate numeric and categorical columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    categorical_cols = df.select_dtypes(include=['object']).columns
    
    # Handle missing values
    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())
    df[categorical_cols] = df[categorical_cols].fillna('None')
    
    # Encode categorical variables
    df = pd.get_dummies(df)
    
    return df


data = preprocess_data(data)

# Split features and target
X = data.drop('SalePrice', axis=1)
y = data['SalePrice']

# Save feature names for later use
feature_names = X.columns
joblib.dump(feature_names, 'feature_names.joblib')

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Feature Scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train Model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate Model
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse}")

# Save the model and scaler
joblib.dump(model, 'model.joblib')
joblib.dump(scaler, 'scaler.joblib')

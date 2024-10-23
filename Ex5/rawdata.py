import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler

# Load the data
data = pd.read_csv("Ex5/raw_data.csv")

# Basic data exploration
print(data.info())
print("\nSample data:")
print(data.head())

# Prepare the data
X = data.drop(['price', 'car_ID', 'CarName'], axis=1)  # Remove non-numeric columns
y = data['price']

# Handle categorical variables
X = pd.get_dummies(X, drop_first=True)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

# Create and train the model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train_scaled, y_train)

# Function to make predictions
def predict_price(features):
    features_encoded = pd.get_dummies(features, drop_first=True)
    features_encoded = features_encoded.reindex(columns=X.columns, fill_value=0)
    features_scaled = scaler.transform(features_encoded)
    return model.predict(features_scaled)[0]

# Example usage
example_features = pd.DataFrame({
    'symboling': [3],
    'fueltype': ['gas'],
    'aspiration': ['std'],
    'doornumber': ['four'],
    'carbody': ['sedan'],
    'drivewheel': ['fwd'],
    'enginelocation': ['front'],
    'wheelbase': [102.4],
    'carlength': [175.6],
    'carwidth': [66.5],
    'carheight': [54.9],
    'curbweight': [2414],
    'enginetype': ['ohc'],
    'cylindernumber': ['four'],
    'enginesize': [122],
    'fuelsystem': ['mpfi'],
    'boreratio': [3.31],
    'stroke': [3.54],
    'compressionratio': [8.7],
    'horsepower': [92],
    'peakrpm': [4200],
    'citympg': [27],
    'highwaympg': [32]
}, index=[0])

predicted_price = predict_price(example_features)
print(f"Predicted price: ${predicted_price:.2f}")
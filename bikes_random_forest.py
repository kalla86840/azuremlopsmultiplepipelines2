import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import joblib
import os

# Load dataset
df = pd.read_csv("data/bikes.csv")
X = df.iloc[:, :-1]
y = df.iloc[:, -1]

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

# Train model
model = RandomForestRegressor()
model.fit(X_train, y_train)

# Save model


import os
os.makedirs('models/bikes/random_forest', exist_ok=True)
import joblib
joblib.dump(model, 'models/bikes/random_forest/model.joblib')

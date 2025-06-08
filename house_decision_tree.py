import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
import joblib
import os

# Load dataset
df = pd.read_csv("data/house.csv")
X = df.iloc[:, :-1]
y = df.iloc[:, -1]

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

# Train model
model = DecisionTreeRegressor()
model.fit(X_train, y_train)

# Save model


import os
os.makedirs('models/house/decision_tree', exist_ok=True)
import joblib
joblib.dump(model, 'models/house/decision_tree/model.joblib')

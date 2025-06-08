import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import joblib
import os

# Load dataset
import os
data_path = os.path.join(os.path.dirname(__file__), "../../data/house.csv")
df = pd.read_csv(data_path)
X = df.iloc[:, :-1]
y = df.iloc[:, -1]

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict on test set
y_pred = model.predict(X_test)

# Save ground truth and predictions
import numpy as np
np.savetxt("y_true_house_LinearRegression.txt", y_test.values, fmt="%.6f")
np.savetxt("y_pred_house_LinearRegression.txt", y_pred, fmt="%.6f")


# Save model

joblib.dump(model, "model.joblib")
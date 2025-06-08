import os
from sklearn.datasets import load_diabetes
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split

# Sample synthetic regression
X, y = load_diabetes(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

model = DecisionTreeRegressor()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

# Save predictions and ground truth
y_true_path = os.path.abspath("y_true_cars_RandomForest.txt")
y_pred_path = os.path.abspath("y_pred_cars_RandomForest.txt")

with open(y_true_path, "w") as f:
    for val in y_test:
        f.write(str(val) + "\n")

with open(y_pred_path, "w") as f:
    for val in y_pred:
        f.write(str(val) + "\n")

print(f"Wrote: {y_true_path} and {y_pred_path}")
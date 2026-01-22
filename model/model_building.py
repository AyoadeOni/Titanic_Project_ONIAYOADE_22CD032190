import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

# ----------------------------------
# 1. Load Dataset
# ----------------------------------
data = pd.read_csv("train.csv")

# ----------------------------------
# 2. Feature Selection
# ----------------------------------
features = ["Pclass", "Sex", "Age", "Fare", "Embarked"]
X = data[features]
y = data["Survived"]

# ----------------------------------
# 3. Handle Missing Values
# ----------------------------------
X["Age"] = X["Age"].fillna(X["Age"].median())
X["Embarked"] = X["Embarked"].fillna(X["Embarked"].mode()[0])

# ----------------------------------
# 4. Encode Categorical Variables
# ----------------------------------
X["Sex"] = X["Sex"].map({"male": 0, "female": 1})
X["Embarked"] = X["Embarked"].map({"S": 0, "C": 1, "Q": 2})

# ----------------------------------
# 5. Train-Test Split
# ----------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ----------------------------------
# 6. Feature Scaling
# ----------------------------------
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ----------------------------------
# 7. Train Model (Logistic Regression)
# ----------------------------------
model = LogisticRegression()
model.fit(X_train_scaled, y_train)

# ----------------------------------
# 8. Evaluation
# ----------------------------------
y_pred = model.predict(X_test_scaled)
print("Classification Report:")
print(classification_report(y_test, y_pred))

# ----------------------------------
# 9. Save Model & Scaler
# ----------------------------------
joblib.dump(model, "model/titanic_survival_model.pkl")
joblib.dump(scaler, "model/scaler.pkl")

print("Model and scaler saved successfully.")

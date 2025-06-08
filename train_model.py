import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import joblib

# 1. Load the dataset
df = pd.read_csv("WA_Fn-UseC_-Telco-Customer-Churn.csv.xls")

# 2. Drop customerID and handle target column
df.drop("customerID", axis=1, inplace=True)
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
df.dropna(inplace=True)

# 3. Convert target
df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})

# 4. One-hot encode categorical features
df = pd.get_dummies(df)

# 5. Split data
X = df.drop('Churn', axis=1)
y = df['Churn']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 6. Train a tuned Random Forest
rf = RandomForestClassifier(class_weight='balanced', random_state=42)
param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [5, 10, 15]
}
grid = GridSearchCV(rf, param_grid, cv=5, scoring='f1', n_jobs=-1)
grid.fit(X_train, y_train)
best_rf = grid.best_estimator_

# 7. Evaluate
y_pred = best_rf.predict(X_test)
print("Classification Report:")
print(classification_report(y_test, y_pred))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# 8. Save the model & columns
joblib.dump(best_rf, 'churn_model.pkl')
joblib.dump(X.columns.tolist(), 'columns.pkl')
print("✅ Model and columns saved successfully.")

import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, roc_curve, classification_report, auc

# Load model & columns
model = joblib.load('churn_model.pkl')
columns = joblib.load('columns.pkl')

# Load and prepare the dataset
import pandas as pd
df = pd.read_csv("WA_Fn-UseC_-Telco-Customer-Churn.csv.xls")
df.drop("customerID", axis=1, inplace=True)
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
df.dropna(inplace=True)
df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})
df = pd.get_dummies(df)
X = df.drop('Churn', axis=1)
y = df['Churn']

# Match columns (in case of extra dummy vars)
X = X[columns]

# Split data
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Get prediction probabilities
y_scores = model.predict_proba(X_test)[:, 1]

# 1. Precision-Recall vs Threshold Plot
precision, recall, thresholds = precision_recall_curve(y_test, y_scores)
plt.figure(figsize=(8, 5))
plt.plot(thresholds, precision[:-1], label="Precision", color="blue")
plt.plot(thresholds, recall[:-1], label="Recall", color="green")
plt.xlabel("Threshold")
plt.ylabel("Score")
plt.title("Precision & Recall vs Threshold")
plt.legend()
plt.grid(True)
plt.show()

# 2. ROC Curve
fpr, tpr, roc_thresholds = roc_curve(y_test, y_scores)
roc_auc = auc(fpr, tpr)
plt.figure(figsize=(6, 6))
plt.plot(fpr, tpr, label=f"ROC Curve (AUC = {roc_auc:.2f})", color='darkorange')
plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.grid(True)
plt.show()

# 3. Evaluate at a Custom Threshold
threshold = 0.60  # Try changing this to 0.4, 0.5, 0.7
y_custom_pred = (y_scores > threshold).astype(int)
print(f"Classification Report @ threshold = {threshold}")
print(classification_report(y_test, y_custom_pred))

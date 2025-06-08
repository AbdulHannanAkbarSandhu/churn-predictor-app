# Customer Churn Predictor App

This interactive Streamlit web application predicts whether a telecom customer is likely to churn based on their account info and service usage. The model behind it is a Random Forest Classifier trained on the IBM Telco Customer Churn dataset, optimized for real-world use with a tunable threshold slider, clean UI, and model evaluation visuals like ROC and precision-recall curves.

---

## Project Overview

- Predicts customer churn with a trained ML model
- Threshold slider (0.50–0.65) to adjust sensitivity
- Optional dark mode
- ROC curve and precision-recall visualization included
- One-click deployment with `streamlit run churnapp.py`

---

## Project Structure

├── churnapp.py # Streamlit app UI
├── train_model.py # Model training script
├── evaluate_thresholds.py # Plots thresholds & ROC curve
├── churn_model.pkl # Trained Random Forest model
├── columns.pkl # Input feature column list
├── WA_Fn-UseC_-Telco...xls # Dataset (public from IBM)
├── Figure_1.png # ROC curve
├── requirements.txt # Python packages
└── README.md # This file


---

## How to Run the App

## 1. Clone the Repository
```bash
git clone https://github.com/YOUR_USERNAME/churn-predictor-app.git
cd churn-predictor-app
```
## 2. Set Up Virtual Environment
```bash python3 -m venv venv
source venv/bin/activate
```
## 3.Install Dependencies
```bash
pip install -r requirements.txt
```
### 4.Launch the App
```bash
streamlit run churnapp.py
```
## Model Performance

Model: Random Forest Classifier

ROC AUC Score: 0.83

Best Threshold Range: 0.60 – 0.65

This model prioritizes recall, making it suitable for use cases where catching potential churners is more important than a few false positives.

## ROC Curve
 Figure_1.png

## Dataset Used

Source: IBM Sample Dataset 

File: WA_Fn-UseC_-Telco-Customer-Churn.csv.xls

Records: 7,043 customers

Features: Gender, Tenure, Charges, Contract Type, and more

## App Features

Dropdowns, sliders, and form inputs for:

Gender

Senior Citizen

Partner


Dependents

Tenure

Monthly Charges

Total Charges

Contract Type

### Live Prediction using:
churn_model.pkl

Encoded using columns.pkl

Threshold slider lets you adjust the prediction cutoff

Optional dark mode UI

Gradient "A.S" logo rendered through custom CSS

## Future Enhancements

#### Add SHAP value explainability
#### Support CSV batch prediction
#### Deploy on Streamlit Cloud or Hugging Face Spaces
#### Add user login and prediction logging

## Author

### Abdul Hannan Sandhu
Focused on building meaningful and interpretable machine learning applications

### License

This project is intended for educational and demonstration purposes only.
The dataset is publicly available from IBM.
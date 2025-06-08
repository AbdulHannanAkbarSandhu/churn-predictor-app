import streamlit as st
import joblib
import numpy as np

# Load model and columns
model = joblib.load("churn_model.pkl")
columns = joblib.load("columns.pkl")

st.set_page_config(page_title="Churn Predictor", layout="centered")

# Persistent dark mode toggle
if "dark_mode" not in st.session_state:
    st.session_state.dark_mode = False

dark_mode = st.checkbox("Enable Dark Mode", value=st.session_state.dark_mode)
st.session_state.dark_mode = dark_mode

# Apply simple CSS
if dark_mode:
    st.markdown(
        """
        <style>
            html, body, [class*="st-"] {
                background-color: #121212;
                color: white;
            }
        </style>
        """,
        unsafe_allow_html=True
    )
else:
    st.markdown(
        """
        <style>
            html, body, [class*="st-"] {
                background-color: white;
                color: black;
            }
        </style>
        """,
        unsafe_allow_html=True
    )

# Logo and title styling
st.markdown("""
    <style>
        @keyframes fade {
            0%,100% { opacity: 0.9; transform: scale(1); }
            50% { opacity: 1; transform: scale(1.08); }
        }

        .logo-container {
            display: flex;
            align-items: center;
            margin-bottom: 20px;
            gap: 20px;
        }

        .logo-text {
            font-size: 88px;
            font-weight: bold;
            background: linear-gradient(135deg, #00c6ff, #0072ff);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            animation: fade 3s infinite;
            margin: 0;
        }

        .app-title {
            font-size: 24px;
            font-weight: 600;
            margin: 0;
        }
    </style>

    <div class="logo-container">
        <div class="logo-text">A.S</div>
        <div class="app-title">Customer Churn Prediction App</div>
    </div>
""", unsafe_allow_html=True)

st.markdown("Enter customer details to predict churn probability.")

# Input fields
gender = st.selectbox("Gender", ['Male', 'Female'])
senior = st.selectbox("Senior Citizen", ['Yes', 'No'])
partner = st.selectbox("Has a Partner?", ['Yes', 'No'])
dependents = st.selectbox("Has Dependents?", ['Yes', 'No'])
tenure = st.slider("Tenure (Months)", 0, 72, 12)
monthly = st.slider("Monthly Charges", 0.0, 150.0, 70.0)
total = st.slider("Total Charges", 0.0, 10000.0, 2000.0)
contract = st.selectbox("Contract Type", ['Month-to-month', 'One year', 'Two year'])

threshold = st.slider(
    "Set churn prediction threshold",
    min_value=0.5,
    max_value=0.65,
    value=0.6,
    step=0.005
)

# Encode user input
input_dict = {
    'gender': 1 if gender == 'Male' else 0,
    'SeniorCitizen': 1 if senior == 'Yes' else 0,
    'Partner': 1 if partner == 'Yes' else 0,
    'Dependents': 1 if dependents == 'Yes' else 0,
    'tenure': tenure,
    'MonthlyCharges': monthly,
    'TotalCharges': total,
    'PaperlessBilling': 1
}

# One-hot for Contract
for col in columns:
    if 'Contract_' in col:
        input_dict[col] = 1 if contract in col else 0
    elif col not in input_dict:
        input_dict[col] = 0

# Format and predict
input_array = np.array([input_dict[col] for col in columns]).reshape(1, -1)

if st.button("Predict"):
    proba = model.predict_proba(input_array)[0][1]
    prediction = 1 if proba > threshold else 0

    st.markdown(f"**Churn Probability:** `{proba:.3f}`")
    st.markdown(f"**Prediction Threshold:** `{threshold:.3f}`")

    if prediction == 1:
        st.error("This customer is likely to churn.")
    else:
        st.success("This customer is likely to stay.")

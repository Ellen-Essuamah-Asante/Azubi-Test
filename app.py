import streamlit as st
import pandas as pd
import numpy as np
import joblib


# Load your trained model
model = joblib.load("xgb_model.pkl")

st.title("Bank Marketing Prediction App")
st.write("Predict whether a customer will subscribe to a term deposit.")

# --- Categorical Mappings ---

job_options = {
    0: 'admin.', 1: 'blue-collar', 2: 'entrepreneur', 3: 'housemaid',
    4: 'management', 5: 'retired', 6: 'self-employed', 7: 'services',
    8: 'student', 9: 'technician', 10: 'unemployed', 11: 'unknown'
}

marital_options = {
    0: 'divorced', 1: 'married', 2: 'single'
}

education_options = {
    0: 'primary', 1: 'secondary', 2: 'tertiary', 3: 'unknown'
}

default_options = {
    0: 'no', 1: 'yes'
}

# --- Streamlit UI ---

job = st.selectbox("Job", list(job_options.values()))
marital = st.selectbox("Marital Status", list(marital_options.values()))
education = st.selectbox("Education", list(education_options.values()))
default = st.selectbox("Default", list(default_options.values()))
balance = st.number_input("Balance", min_value=0)
housing = st.selectbox("Housing Loan", ['yes', 'no'])
loan = st.selectbox("Personal Loan", ['yes', 'no'])

# --- Encode selections back to numerical values ---

job_encoded = list(job_options.keys())[list(job_options.values()).index(job)]
marital_encoded = list(marital_options.keys())[list(marital_options.values()).index(marital)]
education_encoded = list(education_options.keys())[list(education_options.values()).index(education)]
default_encoded = list(default_options.keys())[list(default_options.values()).index(default)]
housing_encoded = 1 if housing == 'yes' else 0
loan_encoded = 1 if loan == 'yes' else 0

# --- Prepare features for model ---
features = np.array([[job_encoded, marital_encoded, education_encoded, default_encoded,
                      balance, housing_encoded, loan_encoded]])

# --- Prediction ---
if st.button("Predict"):
    prediction = model.predict(features)[0]
    st.success("Subscribed ✅" if prediction == 1 else "Not Subscribed ❌")


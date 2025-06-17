import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load your trained model
model = joblib.load("xgb_model.pkl")

st.title("Bank Marketing Prediction App")
st.write("Predict whether a customer will subscribe to a term deposit.")

# --- Mappings for Categorical Fields ---
job_options = {
    0: 'admin.', 1: 'blue-collar', 2: 'entrepreneur', 3: 'housemaid',
    4: 'management', 5: 'retired', 6: 'self-employed', 7: 'services',
    8: 'student', 9: 'technician', 10: 'unemployed', 11: 'unknown'
}

marital_options = {0: 'divorced', 1: 'married', 2: 'single'}
education_options = {0: 'primary', 1: 'secondary', 2: 'tertiary', 3: 'unknown'}
default_options = {0: 'no', 1: 'yes'}
housing_options = {'yes': 1, 'no': 0}
loan_options = {'yes': 1, 'no': 0}
contact_options = {0: 'cellular', 1: 'telephone', 2: 'unknown'}
month_options = {0: 'jan', 1: 'feb', 2: 'mar', 3: 'apr', 4: 'may', 5: 'jun',
                 6: 'jul', 7: 'aug', 8: 'sep', 9: 'oct', 10: 'nov', 11: 'dec'}
poutcome_options = {0: 'failure', 1: 'other', 2: 'success', 3: 'unknown'}
age_group_options = {0: 'young', 1: 'adult', 2: 'senior'}
balance_group_options = {0: 'low', 1: 'medium', 2: 'high'}
contacted_options = {'yes': 1, 'no': 0}

# --- UI Inputs ---
job = st.selectbox("Job", list(job_options.values()))
marital = st.selectbox("Marital Status", list(marital_options.values()))
education = st.selectbox("Education", list(education_options.values()))
default = st.selectbox("Credit in Default?", list(default_options.values()))
balance = st.number_input("Balance", min_value=0)
housing = st.selectbox("Housing Loan", housing_options.keys())
loan = st.selectbox("Personal Loan", loan_options.keys())
contact = st.selectbox("Contact Method", list(contact_options.values()))
day = st.slider("Day of Month Contacted", 1, 31, 15)
month = st.selectbox("Month of Contact", list(month_options.values()))
duration = st.number_input("Last Contact Duration (sec)", min_value=0)
campaign = st.number_input("Number of Contacts This Campaign", min_value=0)
previous = st.number_input("Number of Previous Contacts", min_value=0)
poutcome = st.selectbox("Previous Outcome", list(poutcome_options.values()))
was_previously_contacted = st.selectbox("Previously Contacted?", contacted_options.keys())
age_group = st.selectbox("Age Group", list(age_group_options.values()))
balance_group = st.selectbox("Balance Group", list(balance_group_options.values()))

# --- Encode Inputs ---
features = np.array([[
    list(job_options.keys())[list(job_options.values()).index(job)],
    list(marital_options.keys())[list(marital_options.values()).index(marital)],
    list(education_options.keys())[list(education_options.values()).index(education)],
    list(default_options.keys())[list(default_options.values()).index(default)],
    balance,
    housing_options[housing],
    loan_options[loan],
    list(contact_options.keys())[list(contact_options.values()).index(contact)],
    day,
    list(month_options.keys())[list(month_options.values()).index(month)],
    duration,
    campaign,
    previous,
    list(poutcome_options.keys())[list(poutcome_options.values()).index(poutcome)],
    contacted_options[was_previously_contacted],
    list(age_group_options.keys())[list(age_group_options.values()).index(age_group)],
    list(balance_group_options.keys())[list(balance_group_options.values()).index(balance_group)],
]])

# --- Predict ---
if st.button("Predict"):
    prediction = model.predict(features)[0]
    st.success("✅ Subscribed!" if prediction == 1 else "❌ Not Subscribed.")


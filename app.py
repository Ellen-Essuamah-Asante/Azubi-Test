import streamlit as st
import pandas as pd
import numpy as np
import joblib
import pickle

# --- Load your trained model ---
with open("xgb_model.pkl", "rb") as f:
    model = pickle.load(f)

# --- Define mappings for categorical variables ---
job_map = {
    "Admin.": 0, "Blue-Collar": 1, "Entrepreneur": 2, "Housemaid": 3, "Management": 4,
    "Retired": 5, "Self-Employed": 6, "Services": 7, "Student": 8, "Technician": 9,
    "Unemployed": 10, "Unknown": 11
}

marital_map = {"Married": 0, "Single": 1, "Divorced": 2}
education_map = {"Primary": 0, "Secondary": 1, "Tertiary": 2, "Unknown": 3}
yes_no_map = {"Yes": 1, "No": 0}
contact_map = {"Cellular": 0, "Telephone": 1, "Unknown": 2}
month_map = {
    "January": 0, "February": 1, "March": 2, "April": 3, "May": 4, "June": 5,
    "July": 6, "August": 7, "September": 8, "October": 9, "November": 10, "December": 11
}
poutcome_map = {"Failure": 0, "Other": 1, "Success": 2, "Unknown": 3}
age_group_map = {"Youth": 0, "Adult": 1, "Senior": 2, "Elder": 3, "Unknown": 4}
balance_group_map = {"Low": 0, "Medium": 1, "High": 2, "Very High": 3, "Unknown": 4}

# --- Streamlit App ---
st.title("🔮 Bank Subscription Prediction App")

st.header("Enter Customer Information")

# Dropdowns with labels
job = st.selectbox("Job", list(job_map.keys()))
marital = st.selectbox("Marital Status", list(marital_map.keys()))
education = st.selectbox("Education Level", list(education_map.keys()))
default = st.selectbox("Has Credit in Default?", list(yes_no_map.keys()))
housing = st.selectbox("Has Housing Loan?", list(yes_no_map.keys()))
loan = st.selectbox("Has Personal Loan?", list(yes_no_map.keys()))
contact = st.selectbox("Contact Method", list(contact_map.keys()))
month = st.selectbox("Last Contact Month", list(month_map.keys()))
poutcome = st.selectbox("Previous Outcome", list(poutcome_map.keys()))
was_previously_contacted = st.selectbox("Was Previously Contacted?", list(yes_no_map.keys()))
age_group = st.selectbox("Age Group", list(age_group_map.keys()))
balance_group = st.selectbox("Balance Group", list(balance_group_map.keys()))

# Numeric Inputs
balance = st.number_input("Account Balance", value=0)
day = st.number_input("Last Contact Day", min_value=1, max_value=31, value=15)
duration = st.number_input("Last Contact Duration (seconds)", value=100)
campaign = st.number_input("Number of Contacts During Campaign", value=1)
previous = st.number_input("Number of Previous Contacts", value=0)

# --- Encode selected values ---
input_data = pd.DataFrame([{
    "job": job_map[job],
    "marital": marital_map[marital],
    "education": education_map[education],
    "default": yes_no_map[default],
    "balance": balance,
    "housing": yes_no_map[housing],
    "loan": yes_no_map[loan],
    "contact": contact_map[contact],
    "day": day,
    "month": month_map[month],
    "duration": duration,
    "campaign": campaign,
    "previous": previous,
    "poutcome": poutcome_map[poutcome],
    "was_previously_contacted": yes_no_map[was_previously_contacted],
    "age_group": age_group_map[age_group],
    "balance_group": balance_group_map[balance_group]
}])

# --- Predict ---
if st.button("Predict"):
    prediction = model.predict(input_data)[0]
    if prediction == 1:
        st.success("✅ The client is likely to **subscribe**.")
    else:
        st.warning("❌ The client is **not likely to subscribe**.")


import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import joblib

st.set_page_config(page_title="Patient Progress Dashboard", layout="wide")
st.title("🏥 AI-Based Patient Progress Dashboard")

# Load data and model
df = pd.read_csv("data/patients_data.csv")
model = joblib.load("ai_status_predictor.pkl")

# Preprocess for vitals
df['Systolic'] = df['Blood_Pressure'].apply(lambda x: int(x.split('/')[0]))
df['Diastolic'] = df['Blood_Pressure'].apply(lambda x: int(x.split('/')[1]))
df['Med_Adherence_Code'] = df['Medication_Adherence'].map({'Yes': 1, 'No': 0})

# Sidebar filter
patient_ids = df['Patient_ID'].unique()
selected_patients = st.sidebar.multiselect("Select Patients", patient_ids, default=patient_ids)
filtered_df = df[df['Patient_ID'].isin(selected_patients)]

# Status chart
st.subheader("📊 Health Status Distribution")
status_count = filtered_df['AI_Status'].value_counts()
st.bar_chart(status_count)

# Average vitals
st.subheader("📈 Average Vitals by AI Status")
avg_vitals = filtered_df.groupby('AI_Status')[['Heart_Rate', 'Oxygen_Level', 'Temperature', 'Systolic', 'Diastolic']].mean()
st.dataframe(avg_vitals.style.highlight_max(axis=0))

# Line chart
st.subheader("🧾 Vital Trends Over Time")
selected_patient = st.selectbox("Select Patient for Trend", patient_ids)
patient_data = df[df['Patient_ID'] == selected_patient].sort_values("Date")
st.line_chart(patient_data.set_index("Date")[['Heart_Rate', 'Oxygen_Level', 'Temperature']])

# AI Prediction Section
st.subheader("🤖 Predict AI Health Status")
with st.form("predict_form"):
    hr = st.number_input("Heart Rate", value=85)
    sys = st.number_input("Systolic BP", value=120)
    dia = st.number_input("Diastolic BP", value=80)
    ox = st.number_input("Oxygen Level", value=98.0)
    temp = st.number_input("Temperature (°F)", value=98.6)
    med = st.radio("Medication Adherence", ['Yes', 'No'])
    submit = st.form_submit_button("Predict Status")

    if submit:
        med_code = 1 if med == 'Yes' else 0
        input_df = pd.DataFrame([{
            'Heart_Rate': hr,
            'Systolic': sys,
            'Diastolic': dia,
            'Oxygen_Level': ox,
            'Temperature': temp,
            'Med_Adherence_Code': med_code
        }])
        prediction = model.predict(input_df)[0]
        status_map = {0: "Stable", 1: "At Risk", 2: "Critical"}
        st.success(f"🧠 Predicted AI Status: **{status_map[prediction]}**")

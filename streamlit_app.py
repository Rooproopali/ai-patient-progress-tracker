
import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

st.set_page_config(page_title="Patient Progress Dashboard", layout="wide")
st.title("🏥 AI-Based Patient Progress Dashboard")

# Load the dataset
df = pd.read_csv("data/patients_data.csv")

# Data cleaning
df['Systolic'] = df['Blood_Pressure'].apply(lambda x: int(x.split('/')[0]))
df['Diastolic'] = df['Blood_Pressure'].apply(lambda x: int(x.split('/')[1]))

# Sidebar filter
patient_ids = df['Patient_ID'].unique()
selected_patients = st.sidebar.multiselect("Select Patients", patient_ids, default=patient_ids)

filtered_df = df[df['Patient_ID'].isin(selected_patients)]

# AI Status count
st.subheader("📊 Health Status Distribution")
status_count = filtered_df['AI_Status'].value_counts()
st.bar_chart(status_count)

# Average vitals by AI_Status
st.subheader("📈 Average Vitals by AI Status")
avg_vitals = filtered_df.groupby('AI_Status')[['Heart_Rate', 'Oxygen_Level', 'Temperature', 'Systolic', 'Diastolic']].mean()
st.dataframe(avg_vitals.style.highlight_max(axis=0))

# Line chart for patient trends
st.subheader("🧾 Vital Trends Over Time")
selected_patient = st.selectbox("Select Patient for Trend", patient_ids)
patient_data = df[df['Patient_ID'] == selected_patient].sort_values("Date")

st.line_chart(patient_data.set_index("Date")[['Heart_Rate', 'Oxygen_Level', 'Temperature']])

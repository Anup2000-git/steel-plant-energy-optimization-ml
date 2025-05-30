import logging
import streamlit as st
import pickle
import pandas as pd
import numpy as np
from PIL import Image

# Suppress logging warnings
logging.getLogger('streamlit.runtime.scriptrunner.script_run_context').setLevel(logging.ERROR)

# Load saved objects
with open("D:\\360DigiTMG Date 28Aug\\Project_360DigiTMG_HYD_26_02_25__02\\model building\\best_model.pkl", 'rb') as f:
    model = pickle.load(f)

with open("D:\\360DigiTMG Date 28Aug\\Project_360DigiTMG_HYD_26_02_25__02\\model building\\scaler.pkl", 'rb') as f:
    scaler = pickle.load(f)

with open("D:\\360DigiTMG Date 28Aug\\Project_360DigiTMG_HYD_26_02_25__02\\model building\\label_encoders.pkl", 'rb') as f:
    label_encoders = pickle.load(f)

with open("D:\\360DigiTMG Date 28Aug\\Project_360DigiTMG_HYD_26_02_25__02\\model building\\feature_names.pkl", 'rb') as f:
    feature_names = pickle.load(f)

# Load an image banner
st.image("D:\\360DigiTMG Date 28Aug\\Project_360DigiTMG_HYD_26_02_25__02\\model building\\images.jpg"", use_column_width=True")

# App title and description
st.title('Steel Manufacturing Optimization - Cluster Prediction')
st.markdown("### Optimize production efficiency by predicting the best steel grade cluster.")

# Layout the input fields
col1, col2 = st.columns(2)

with col1:
    production = st.number_input('Production (MT)', value=500.0)
    energy = st.number_input('ENERGY (Energy Consumption)', value=1000.0)
with col2:
    cycle_time = st.number_input('TT_TIME (Total Cycle Time Including Breakdown)', value=200.0)

st.markdown("---")

# Collect categorical inputs
categorical_inputs = {}
st.subheader("Select Categorical Features")
for col, le in label_encoders.items():
    options = le.classes_.tolist()
    selected = st.selectbox(f'{col}', options)
    categorical_inputs[col] = selected

# Prepare input DataFrame
input_data = pd.DataFrame(columns=feature_names)
input_data.loc[0] = 0  # Fill all columns with default zero

# Fill numerical inputs
input_data.at[0, 'Production (MT)'] = production
input_data.at[0, 'ENERGY (Energy Consumption)'] = energy
input_data.at[0, 'TT_TIME (Total Cycle Time Including Breakdown)'] = cycle_time

# Encode categorical inputs
for col, le in label_encoders.items():
    input_data.at[0, col] = le.transform([categorical_inputs[col]])[0]

# Scale numeric columns
numeric_cols = input_data.select_dtypes(include=[np.number]).columns
input_data[numeric_cols] = scaler.transform(input_data[numeric_cols])

# Prediction button
if st.button('Predict Steel Grade Cluster'):
    prediction = model.predict(input_data)
    st.success(f"### Predicted Steel Grade Cluster: {prediction[0]}")
    
    # Display input values for confirmation
    st.subheader("Input Summary")
    st.dataframe(input_data)

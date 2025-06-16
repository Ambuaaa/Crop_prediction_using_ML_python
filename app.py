import streamlit as st
import numpy as np
import pandas as pd
import pickle

# Load your model and preprocessor
dtr = pickle.load(open('dtr.pkl', 'rb'))
preprocessor = pickle.load(open('preprocessor.pkl', 'rb'))

st.title("🌾 Crop Yield Prediction App")

# Input fields
Year = st.text_input("Year")
average_rain_fall_mm_per_year = st.number_input("Average Rainfall (mm/year)")
pesticides_tonnes = st.number_input("Pesticide Usage (tonnes)")
avg_temp = st.number_input("Average Temperature (°C)")
Area = st.text_input("Area (State)")
Item = st.text_input("Crop Name")

if st.button("Predict"):
    if not all([Year, Area, Item]):
        st.warning("Please fill all text inputs")
    else:
        features = np.array([[Year, average_rain_fall_mm_per_year, pesticides_tonnes, avg_temp, Area, Item]])
        transformed_features = preprocessor.transform(features)
        predicted_value = dtr.predict(transformed_features).reshape(1, -1)

        st.success(f"✅ Predicted Crop Yield: {predicted_value[0][0]:.2f} hg/ha")


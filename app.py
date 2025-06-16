import streamlit as st
import numpy as np
import pickle

# Title and layout
st.set_page_config(page_title="Crop Yield Predictor", layout="centered")
st.markdown("<h1 style='text-align: center; color: green;'>🌾 Crop Yield Prediction Per Country</h1>", unsafe_allow_html=True)
st.write("---")

# Load model and preprocessor
@st.cache(allow_output_mutation=True)
def load_objects():
    with open("dtr.pkl", "rb") as f:
        model = pickle.load(f)
    with open("preprocessor.pkl", "rb") as f:
        preprocessor = pickle.load(f)
    return model, preprocessor

dtr, preprocessor = load_objects()

# Input form
with st.form("prediction_form"):
    Year = st.number_input("Year", step=1, value=2024)
    rainfall = st.number_input("Average Rainfall (mm/year)", step=0.1)
    pesticides = st.number_input("Pesticides (tonnes)", step=0.1)
    avg_temp = st.number_input("Average Temperature (°C)", step=0.1)
    Area = st.text_input("Area (State)")
    Item = st.text_input("Crop Name")
    submit = st.form_submit_button("Predict")

if submit:
    if not Area or not Item:
        st.warning("Please fill all text fields.")
    else:
        features = np.array([[Year, rainfall, pesticides, avg_temp, Area, Item]])
        transformed = preprocessor.transform(features)
        pred = dtr.predict(transformed)[0]
        st.success(f"✅ Predicted Crop Yield: {pred:.2f} hg/ha")

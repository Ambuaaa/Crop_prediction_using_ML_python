import streamlit as st
import numpy as np
import pickle

st.set_page_config(page_title="Crop Yield Predictor", layout="centered")
st.markdown("""
    <style>
        .main-container {
            background-color: darkkhaki;
            padding: 2rem;
            border-radius: 10px;
        }
        .heading {
            text-align: center;
            color: green;
        }
        .sub-heading {
            text-align: center;
            color: darkgreen;
        }
    </style>
""", unsafe_allow_html=True)

st.markdown("<h1 class='heading'>Crop Yield Prediction Per Country</h1>", unsafe_allow_html=True)
st.write("---")

# Load model and preprocessor using new caching logic
@st.cache_resource
def load_model():
    with open("dtr.pkl", "rb") as f:
        model = pickle.load(f)
    with open("preprocessor.pkl", "rb") as f:
        preprocessor = pickle.load(f)
    return model, preprocessor

dtr, preprocessor = load_model()

with st.container():
    st.markdown("<div class='main-container'>", unsafe_allow_html=True)
    st.markdown("<h2 class='sub-heading'>Crop Features Here</h2>", unsafe_allow_html=True)

    with st.form("predict_form"):
        Year = st.number_input("Year", step=1, value=2024)
        average_rain_fall_mm_per_year = st.number_input("Average Rainfall (mm/year)", step=0.1)
        pesticides_tonnes = st.number_input("Pesticides Used (tonnes)", step=0.1)
        avg_temp = st.number_input("Average Temperature (°C)", step=0.1)
        Area = st.text_input("Area (State or Country)")
        Item = st.text_input("Crop Type")
        submit_button = st.form_submit_button(label='Predict')

    if submit_button:
        if not Area or not Item:
            st.warning("Please fill in all fields.")
        else:
            input_data = np.array([[Year, average_rain_fall_mm_per_year, pesticides_tonnes, avg_temp, Area, Item]])
            transformed_data = preprocessor.transform(input_data)
            prediction = dtr.predict(transformed_data)[0]
            st.success(f"Predicted Yield Production: {prediction:.2f} hg/ha")

    st.markdown("</div>", unsafe_allow_html=True)

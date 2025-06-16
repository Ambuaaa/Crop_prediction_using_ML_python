import streamlit as st
import numpy as np
import pickle

st.set_page_config(page_title="Crop Yield Predictor", layout="centered")

# Use CSS through markdown
st.markdown("""
    <style>
        .main-container {
            background-color: #f5f5dc;
            padding: 2rem;
            border-radius: 10px;
        }
        .heading {
            text-align: center;
            color: #0f5132;
        }
        .sub-heading {
            text-align: center;
            color: #14532d;
        }
        .prediction {
            text-align: center;
            color: #b91c1c;
            font-size: 1.5rem;
            margin-top: 2rem;
        }
    </style>
""", unsafe_allow_html=True)

st.markdown("<h1 class='heading'>Crop Yield Prediction Per Country</h1>", unsafe_allow_html=True)

# Load model and preprocessor using updated caching
@st.cache_resource(show_spinner=False)
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
            st.markdown(f"<div class='prediction'>Predicted Yield Production: {prediction:.2f} hg/ha</div>", unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)

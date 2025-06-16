import streamlit as st
import numpy as np
import pickle

st.set_page_config(page_title="Crop Yield Prediction", layout="centered")

# Load model and preprocessor
@st.cache_resource(show_spinner=False)
def load_models():
    model = pickle.load(open('dtr.pkl', 'rb'))
    preprocessor = pickle.load(open('preprocessor.pkl', 'rb'))
    return model, preprocessor

dtr, preprocessor = load_models()

# CSS styling to match original HTML look
st.markdown("""
    <style>
    body {
        background-color: darkkhaki;
    }
    .container {
        background-color: beige;
        padding: 30px;
        border-radius: 10px;
        margin-top: 30px;
    }
    .title {
        text-align: center;
        color: green;
    }
    .predict {
        text-align: center;
        color: red;
        font-size: 24px;
        margin-top: 20px;
    }
    </style>
""", unsafe_allow_html=True)

# Main Title
st.markdown("<h1 class='title'>Crop Yield Prediction Per Country</h1>", unsafe_allow_html=True)

with st.container():
    st.markdown("<div class='container'>", unsafe_allow_html=True)
    st.markdown("<h2 class='title'>Crop Features Here</h2>", unsafe_allow_html=True)

    with st.form("predict_form"):
        Year = st.number_input("Year", step=1, value=2024)
        average_rain_fall_mm_per_year = st.number_input("average_rain_fall_mm_per_year", step=0.1)
        pesticides_tonnes = st.number_input("pesticides_tonnes", step=0.1)
        avg_temp = st.number_input("avg_temp", step=0.1)
        Area = st.text_input("Area")
        Item = st.text_input("Item")

        submit = st.form_submit_button("Predict")

    if submit:
        if not Area or not Item:
            st.warning("Please fill in all fields.")
        else:
            features = np.array([[Year, average_rain_fall_mm_per_year, pesticides_tonnes, avg_temp, Area, Item]])
            transformed = preprocessor.transform(features)
            predicted_value = dtr.predict(transformed)[0]
            st.markdown(f"<div class='predict'>Predicted Yield Production: {predicted_value:.2f} hg/ha</div>", unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)

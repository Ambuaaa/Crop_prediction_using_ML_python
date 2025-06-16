import streamlit as st
import numpy as np
import pickle

# Load the model and preprocessor
dtr = pickle.load(open("dtr.pkl", "rb"))
preprocessor = pickle.load(open("preprocessor.pkl", "rb"))

# Apply custom CSS to mimic your HTML design
st.markdown("""
    <style>
        body {
            background-color: darkkhaki;
        }
        .main {
            background-color: darkkhaki;
        }
        .container {
            background-color: beige;
            padding: 2rem;
            border-radius: 12px;
            box-shadow: 2px 2px 10px rgba(0,0,0,0.3);
        }
        .title {
            color: green;
            text-align: center;
        }
        .sub-title {
            color: darkred;
            text-align: center;
        }
    </style>
""", unsafe_allow_html=True)

# Page title
st.markdown("<h1 class='title'>Crop Yield Prediction Per Country</h1>", unsafe_allow_html=True)
st.markdown("<div class='container'>", unsafe_allow_html=True)

st.markdown("<h2 class='title'>Crop Features Here</h2>", unsafe_allow_html=True)

# Input fields
Year = st.number_input("Year", step=1, value=2024)
average_rain_fall_mm_per_year = st.number_input("average_rain_fall_mm_per_year", step=0.01)
pesticides_tonnes = st.number_input("pesticides_tonnes", step=0.01)
avg_temp = st.number_input("avg_temp", step=0.01)
Area = st.text_input("Area")
Item = st.text_input("Item")

if st.button("Predict"):
    features = np.array([[Year, average_rain_fall_mm_per_year, pesticides_tonnes, avg_temp, Area, Item]])
    transformed_features = preprocessor.transform(features)
    predicted_value = dtr.predict(transformed_features).reshape(1, -1)
    st.markdown(f"<h2 class='sub-title'>Predicted Yield Production:<br> {predicted_value[0][0]:.2f}</h2>", unsafe_allow_html=True)

st.markdown("</div>", unsafe_allow_html=True)

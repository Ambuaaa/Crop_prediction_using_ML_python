# 🌾 Crop Yield Prediction using Supervised Machine Learning

This project aims to predict **crop yield in advance** by analyzing various environmental and agricultural factors. By using supervised machine learning techniques, the model provides farmers and agricultural planners with insights to choose the best crop and optimize planning for better yield outcomes.

---

## 📌 Objective

To build a machine learning model that can predict crop yield based on:
- 🌍 **Country**
- 🗺️ **State**
- 📅 **Season**
- 🌱 **Crop Type**
- 🌡️ **Average Temperature**
- 🌧️ **Rainfall**
- 🧪 **Pesticide Usage**

---

## ⚙️ Technologies Used

- 🐍 **Python 3**
- 📘 **Pandas, NumPy**
- 📊 **Matplotlib, Seaborn** (for EDA and visualization)
- 🤖 **Scikit-learn** (for machine learning models)
- 💾 **Pickle** (for model saving)
- 🧹 **ColumnTransformer, OneHotEncoder, StandardScaler** (for preprocessing)

---

## 🧠 Models Used

Several supervised learning algorithms were experimented with:
- 🔍 **Linear Regression**
- 🌳 **Decision Tree Regressor**
- 🌲 **Random Forest Regressor**
- 📈 **Gradient Boosting Regressor**

---

## 📊 Dataset Features

The dataset includes the following columns:
- `Country`
- `State`
- `Season`
- `Crop`
- `average_rain_fall_mm_per_year`
- `avg_temp`
- `pesticides_tonnes`
- `hg/ha_yield` (Target variable)

---

## 🛠️ How It Works

1. **Data Cleaning & Preprocessing**
   - Handled missing values
   - Encoded categorical features
   - Scaled numerical features

2. **Train-Test Split**
   - 80% training / 20% testing

3. **Model Training & Evaluation**
   - Used metrics like R² score and RMSE
   - Compared multiple models for best accuracy

4. **Prediction**
   - Once trained, the model predicts the yield (in hg/ha) given the inputs

---

## 💡 Use Case

> Helps farmers and agri-businesses to **plan crop selection** based on predicted yields, improving food security and profitability.

---
## 🚀 Future Improvements

- 🛰️ Integrate real-time weather data APIs
- 📍 Add geolocation-based recommendations
- 🌐 Build a web app using Flask or Streamlit

---

## 🙌 Acknowledgements

Inspired by real-world problems in Indian agriculture and guided by various machine learning tutorials.



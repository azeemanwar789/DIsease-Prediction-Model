import streamlit as st
import pickle
import numpy as np
from sklearn.preprocessing import StandardScaler

# Load the saved models (make sure these are in the same directory as this app)
with open(r"C:\Azeem's Work\IDC Internship\2nd Month\Report-Tasks\Week 5  Day 4 -6 Task Material\logistic_regression_model(Diabetes).pkl", 'rb') as file:
    lr_model = pickle.load(file)

with open(r"C:\Azeem's Work\IDC Internship\2nd Month\Report-Tasks\Week 5  Day 4 -6 Task Material\decision_tree_model(Diabetes).pkl", 'rb') as file:
    dt_model = pickle.load(file)

# Streamlit App Title
st.title('Diabetes Prediction System')

# Input form for the prediction
st.subheader('Enter the following details to predict Diabetes')

# User inputs
pregnancies = st.number_input("Pregnancies", min_value=0)
glucose = st.number_input("Glucose", min_value=0)
blood_pressure = st.number_input("Blood Pressure", min_value=0)
skin_thickness = st.number_input("Skin Thickness", min_value=0)
insulin = st.number_input("Insulin", min_value=0)
bmi = st.number_input("BMI", min_value=0.0)
diabetes_pedigree_function = st.number_input("Diabetes Pedigree Function", min_value=0.0)
age = st.number_input("Age", min_value=0)

# Feature array
features = np.array([[pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, diabetes_pedigree_function, age]])

# Standardizing the input data
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)

# Dropdown for selecting model
model_choice = st.selectbox("Select the model for prediction", ["Logistic Regression", "Decision Tree"])

# Predict button
if st.button("Predict"):
    if model_choice == "Logistic Regression":
        # Prediction using Logistic Regression
        prediction = lr_model.predict(features_scaled)
        st.write(f"Predicted Outcome: {'Positive' if prediction[0] == 1 else 'Negative'}")
    elif model_choice == "Decision Tree":
        # Prediction using Decision Tree
        prediction = dt_model.predict(features_scaled)
        st.write(f"Predicted Outcome: {'Positive' if prediction[0] == 1 else 'Negative'}")


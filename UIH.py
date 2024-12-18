import streamlit as st
import pickle
import numpy as np
from sklearn.preprocessing import StandardScaler

# Load the saved models (make sure these are in the same directory as this app)
with open(r"C:\Azeem's Work\IDC Internship\2nd Month\Report-Tasks\Week 5  Day 4 -6 Task Material\logistic_regression_model(HD).pkl", 'rb') as file:
    lr_model = pickle.load(file)

with open(r"C:\Azeem's Work\IDC Internship\2nd Month\Report-Tasks\Week 5  Day 4 -6 Task Material\decision_tree_model(HD).pkl", 'rb') as file:
    dt_model = pickle.load(file)

# Streamlit App Title
st.title('Heart Disease Prediction System')

# Input form for the prediction
st.subheader('Enter the following details to predict Diabetes')

# User inputs
age = st.number_input("Age", min_value=0)
sex = st.selectbox("Sex", [0, 1])  # 0: Female, 1: Male
cp = st.selectbox("Chest Pain Type", [0, 1, 2, 3])
restbps = st.number_input("Resting Blood Pressure", min_value=0)
chol = st.number_input("Serum Cholesterol", min_value=0)
fbs = st.selectbox("Fasting Blood Sugar", [0, 1])  # 0: Normal, 1: Greater than 120 mg/dl
restecg = st.selectbox("Resting Electrocardiographic Results", [0, 1, 2])
thalach = st.number_input("Maximum Heart Rate Achieved", min_value=0)
exang = st.selectbox("Exercise Induced Angina", [0, 1])  # 0: No, 1: Yes
oldpeak = st.number_input("ST Depression Induced by Exercise", min_value=0.0)
slope = st.selectbox("Slope of Peak Exercise ST Segment", [0, 1, 2])
ca = st.selectbox("Number of Major Vessels Colored by Fluoroscopy", [0, 1, 2, 3])
thal = st.selectbox("Thalassemia", [0, 1, 2, 3])  # 3: Normal, 1: Fixed defect, 2: Reversible defect


# Feature array
features = np.array([[sex, cp, restbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, age, ca, thal]])

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


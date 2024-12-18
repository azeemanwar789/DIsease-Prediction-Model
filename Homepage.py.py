import streamlit as st
import os

# Set the title of the homepage
st.title("Welcome to the Prediction Portal")

# Add some explanation or instructions
st.write("Choose the prediction model you want to use:")

# Create two large buttons for navigation to different apps
col1, col2 = st.columns(2)

# Define the directory path and file names for the apps
base_path = r"C:\Azeem's Work\IDC Internship\2nd Month\Report-Tasks\Week 6  Day 1 to 3 Task"
heart_disease_app = "UIH.py"
diabetes_app = "UID.py"

# Function to run terminal commands
def run_app(app_name):
    # Change directory to the specified base path
    os.chdir(base_path)
    # Run the Streamlit app
    os.system(f'streamlit run {app_name}')

with col1:
    if st.button("Heart Disease Prediction"):
        # Run the Heart Disease Prediction app
        run_app(heart_disease_app)

with col2:
    if st.button("Diabetes Prediction"):
        # Run the Diabetes Prediction app
        run_app(diabetes_app)

# Optionally, add some footer or information at the bottom
st.markdown("---")
st.write("Developed by Azeem Anwar | Machine Learning Predictions")

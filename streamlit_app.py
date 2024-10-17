import streamlit as st
import pandas as pd
import joblib


# Load the trained model and scaler
model = joblib.load('student_alcohol_model.pkl')
scaler = joblib.load('scaler.pkl')  # Ensure the scaler used in training is loaded if scaling was applied

# Streamlit app title
st.title("Student Alcohol Consumption Prediction")

# Description of the app
st.write(
    "This app predicts student alcohol consumption during weekdays based on input data such as age, study time, health, and other factors.")

# Input form for users to input data
st.header("Enter Student Information:")

# Collect user input
age = st.slider('Age', 15, 22, 18)
studytime = st.slider('Study Time (hours/week)', 1, 4, 2)
health = st.slider('Health Status (1 = worst, 5 = best)', 1, 5, 3)
absences = st.number_input('Number of School Absences', min_value=0, value=2)
G1 = st.number_input('Grade 1 (out of 20)', min_value=0, max_value=20, value=12)
G2 = st.number_input('Grade 2 (out of 20)', min_value=0, max_value=20, value=14)

# Optional: You can include other input fields based on your model's features

# When the user clicks the "Predict" button
if st.button('Predict Alcohol Consumption Level'):
    # Create the input data as a DataFrame
    input_data = pd.DataFrame([[age, studytime, health, absences, G1, G2]],
                              columns=['age', 'studytime', 'health', 'absences', 'G1', 'G2'])

    # Scale the input data if scaling was applied during training
    input_scaled = scaler.transform(input_data)

    # Make the prediction
    prediction = model.predict(input_scaled)

    # Display the prediction
    st.write(f"Predicted Alcohol Consumption (Weekday): {prediction[0]}")

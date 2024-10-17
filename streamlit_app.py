import streamlit as st
import pandas as pd
import joblib



# Load the trained model and scaler
with open('student_alcohol_model.pkl', 'rb') as f:
    model = joblib.load(f)

with open('scaler.pkl', 'rb') as f:
    scaler = joblib.load(f)

# Define the feature list exactly as used in training
features = ['age', 'studytime', 'health', 'absences', 'G1', 'G2', 'G3', 'Fedu', 'Medu', 'Pstatus', 'Walc']
# Streamlit app title
st.title("Student Alcohol Consumption Prediction")

# Description of the app
st.write(
    "This app predicts student alcohol consumption during weekdays based on input data such as age, study time, health, and other factors.")

# Input form for users to input data
st.header("Enter Student Information:")

# Collect user input
age = st.slider('Age', 15, 22, 18, 25)
studytime = st.slider('Study Time (hours/week)', 1, 4, 2)
health = st.slider('Health Status (1 = worst, 5 = best)', 1, 5, 3)
absences = st.number_input('Number of School Absences', min_value=0, value=2)
G1 = st.number_input('Grade 1 (out of 20)', min_value=0, max_value=20, value=12)
G2 = st.number_input('Grade 2 (out of 20)', min_value=0, max_value=20, value=14)
G3 = st.number_input('Grade 3 (out of 20)', min_value=0, max_value=20, value=14)
Fedu = st.number_input('Father\'s Education (0-4)', min_value=0, max_value=4)
Medu = st.number_input('Mother\'s Education (0-4)', min_value=0, max_value=4)

# Get Pstatus from the selectbox
Pstatus_input = st.selectbox('Parent\'s Cohabitation Status', ['T', 'A'])

# Convert Pstatus to numerical values
Pstatus = 1 if Pstatus_input == 'T' else 0

Walc = st.number_input('Weekend Alcohol Consumption (1-5)', min_value=1, max_value=5)

# Store the input values in the same order as the features
input_data = [age, studytime, health, absences, G1, G2, G3, Fedu, Medu, Pstatus, Walc]

# Convert the input data into a DataFrame for scaling
input_df = pd.DataFrame([input_data], columns=features)

try:
    # Scale the input data
    input_scaled = scaler.transform(input_df)

    # Make the prediction
    prediction = model.predict(input_scaled)

    # Display the prediction result
    st.write(f"Predicted Alcohol Consumption (Weekday) level: {prediction[0]}")

except Exception as e:
    st.error(f"An error occurred: {e}")

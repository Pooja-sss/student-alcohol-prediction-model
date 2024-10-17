import pandas as pd
import streamlit as st
import joblib
from sklearn.preprocessing import StandardScaler

# Load the trained model and scaler
with open('alcohol_consumption_model.pkl', 'rb') as f:
    model = joblib.load(f)

with open('scaler.pkl', 'rb') as f:
    scaler = joblib.load(f)

# Define the feature list exactly as used in training
features = ['age', 'studytime', 'health', 'absences', 'G1', 'G2', 'G3', 'Fedu', 'Medu', 'Pstatus', 'Walc']

# Streamlit app interface
st.title('Student Alcohol Consumption Prediction')

# Input fields for the user to enter values for each feature
age = st.number_input('Age', min_value=15, max_value=22)
studytime = st.number_input('Study Time (hours)', min_value=1, max_value=4)
health = st.number_input('Health (1-5)', min_value=1, max_value=5)
absences = st.number_input('Absences', min_value=0)
G1 = st.number_input('First Period Grade (G1)', min_value=0, max_value=20)
G2 = st.number_input('Second Period Grade (G2)', min_value=0, max_value=20)
G3 = st.number_input('Final Grade (G3)', min_value=0, max_value=20)
Fedu = st.number_input('Father\'s Education (0-4)', min_value=0, max_value=4)
Medu = st.number_input('Mother\'s Education (0-4)', min_value=0, max_value=4)
Pstatus = st.selectbox('Parent\'s Cohabitation Status', ['T', 'A'])
Walc = st.number_input('Weekend Alcohol Consumption (1-5)', min_value=1, max_value=5)

# Store the input values in the same order as the features
input_data = [age, studytime, health, absences, G1, G2, G3, Fedu, Medu, Pstatus, Walc]

# Convert the input data into a DataFrame for scaling
input_df = pd.DataFrame([input_data], columns=features)

# Scale the input data
input_scaled = scaler.transform(input_df)

# Make the prediction
prediction = model.predict(input_scaled)

# Display the prediction result
st.write(f"Predicted Alcohol Consumption (Weekday) level: {prediction[0]}")

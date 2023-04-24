import streamlit as st
import pandas as pd
import pickle

# Load the model, scaler and encoder
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)
with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

# Create the input form for the user
st.title('Diabetes Mellitus Prediction')
st.write('Please fill in the following details:')
bmi = st.number_input('Baseline BMI (Kg/m2)')
hip = st.number_input('Hip')
weight = st.number_input('Baseline Wt (kg)')
waist = st.number_input('Waist')
sex = st.selectbox('Sex', ['Male', 'Female'])
triglycerides = st.number_input('Triglycerides')
fbs = st.number_input('Baseline FBS')
cholesterol = st.number_input('Total Cholesterol')

# Encode the sex feature
if sex == 'Male':
    sex_encoded = 1
else:
    sex_encoded = 2

# Scale the input data
input_data = pd.DataFrame({'Baseline BMI (Kg/m2)': [bmi], 'hip': [hip], 'baseline Wt (kg)': [weight],
                           'waist': [waist], 'Sex': [sex_encoded], 'triglycerides': [triglycerides],
                           'baseline FBS': [fbs], 'Total cholestrol': [cholesterol]})
scaled_data = scaler.transform(input_data)

# Make the prediction
prediction = model.predict(scaled_data)[0]

# Convert prediction to string output
if prediction == 0:
    result = 'Diabetes Mellitus Obese Class 1'
else:
    result = 'Diabetes Mellitus Obese Class 2'

# Create submit button to display prediction
if st.button('Submit'):
    st.write(f'The predicted result is: {result}')

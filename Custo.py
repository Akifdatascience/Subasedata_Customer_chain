#!/usr/bin/env python
# coding: utf-8

# In[5]:


import tensorflow as tf
import streamlit as st
import numpy as np

# Load the Keras model
model = tf.keras.models.load_model('my_model.hdfs')

# Mapping of locations
location_mapping = {'Los Angeles': 0, 'New York': 1, 'Miami': 2, 'Chicago': 3, 'Houston': 4}

def predict_churn(age, gender, location, subscription_length, monthly_bill, total_usage):
    # Convert gender to numerical value (0 for Female, 1 for Male)
    gender_num = 0 if gender == 'Female' else 1

    # Convert location to numerical value
    location_num = location_mapping.get(location, -1)
    
    if location_num == -1:
        st.error("Invalid location selected.")
        return None

    # Create a feature array
    features = np.array([[age, gender_num, location_num, subscription_length, monthly_bill, total_usage]])

    # Make prediction
    prediction = model.predict(features)

    return prediction

# Streamlit UI
st.title('Customer Churn Prediction')

st.sidebar.header('User Input')

# Collect user input
age = st.sidebar.slider('Age', 18, 100, 30)
gender = st.sidebar.radio('Gender', ['Female', 'Male'])
location = st.sidebar.selectbox('Location', ['Los Angeles', 'New York', 'Miami', 'Chicago', 'Houston'])
subscription_length = st.sidebar.slider('Subscription Length (months)', 1, 24, 12)
monthly_bill = st.sidebar.number_input('Monthly Bill', min_value=0.0, step=1.0)
total_usage = st.sidebar.number_input('Total Usage (GB)', min_value=0.0, step=1.0)

# Predict churn
if st.sidebar.button('Predict Churn'):
    prediction = predict_churn(age, gender, location, subscription_length, monthly_bill, total_usage)
    
    if prediction is not None:
        churn_status = 'Churn' if prediction > 0.5 else 'No Churn'
        st.write(f'Based on the input, the predicted churn status is: {churn_status}')

# Information and tips
st.info("This is a simple Customer Churn Prediction app. Adjust the sliders and input fields on the left sidebar and click the 'Predict Churn' button to see the prediction.")


# In[ ]:





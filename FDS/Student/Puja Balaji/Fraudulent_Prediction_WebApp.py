# -*- coding: utf-8 -*-
"""
Created on Fri Jun  7 16:45:28 2024

@author: Puja
"""
#Importing the libraries
import numpy as np
import pandas as pd
import pickle
import streamlit as st
import os

# Print the current working directory
print("Current working directory:", os.getcwd())

# Loading the models
loaded_models = pickle.load(open('C:/Users/bpuni/Downloads/trained_models.sav', 'rb'))

#Predictive System
def predict_and_determine_fraudulent(loaded_models, new_data_dict):
    # Converting the dictionary to a DataFrame
    new_data = pd.DataFrame(new_data_dict)
    
    # Initializing a dictionary to store predictions
    predictions = {}
    
    # Iterating over each model to make predictions
    for model_name, model in loaded_models.items():
        y_pred = model.predict(new_data)
        predictions[model_name] = y_pred[0]  # Assuming y_pred is a list or array and taking the first element
    
    # Determining if the transaction is fraudulent based on the predictions from all models
    fraudulent_predictions = {}
    for model_name, prediction in predictions.items():
        if prediction == 1:
            fraudulent_predictions[model_name] = 'Fraudulent'
        else:
            fraudulent_predictions[model_name] = 'Not Fraudulent'
    
    return fraudulent_predictions

def main():
    # Giving a title for the Web App
    st.title('Fraudulent Transaction Detection System')
    
    # Getting inputs from user
    customer_id = st.number_input('Enter Customer ID', min_value=1)
    amount = st.number_input('Enter Purchase Amount', min_value=0.0, format="%.2f")
    card_type = st.radio('Select Card Type', ['MasterCard', 'Visa', 'Discover', 'American Express'], index=0)
    location = st.text_input('Enter Location (1 to 50)', '35')
    purchase_category = st.radio('Select Purchase Category', ['Gas Station', 'Online Shopping', 'Travel', 'Retail', 'Groceries', 'Restaurant'], index=2)
    customer_age = st.number_input('Enter Customer Age', min_value=0)
    year = st.number_input('Enter Year', min_value=2020, max_value=2100, value=2024)
    month = st.number_input('Enter Month', min_value=1, max_value=12, value=6)
    day = st.number_input('Enter Day', min_value=1, max_value=31, value=7)

    # Mapping the values for CardType and Purchase Category
    card_type_map = {'MasterCard': 2, 'Visa': 3, 'Discover': 1, 'American Express': 0}
    purchase_category_map = {'Gas Station': 0, 'Online Shopping': 2, 'Travel': 5, 'Retail': 4, 'Groceries': 1, 'Restaurant': 3}
    
    # Prediction result
    predictions_for_all_models = ''
    
    # Creating a button for prediction
    if st.button('Check Transaction'):
        new_data_dict = {
            'customer_id': [customer_id],
            'amount': [amount],
            'customer_age': [customer_age],
            'Year': [year],
            'Month': [month],
            'Day': [day],
            'ct': [card_type_map[card_type]],
            'loc': [int(location)],
            'pc': [purchase_category_map[purchase_category]],
        }
        
        # Printing new_data_dict to verify the input data
        st.write("Input Data:", new_data_dict)
        
        predictions_for_all_models = predict_and_determine_fraudulent(loaded_models, new_data_dict)
        
        # Printing predictions to verify the output
        st.write("Predictions:", predictions_for_all_models)
        
        # Displaying the results
        for model_name, prediction in predictions_for_all_models.items():
            st.success(f"{model_name}: {prediction}")

if __name__ == '__main__':
    main()

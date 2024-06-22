# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
#Importing the libraries
import numpy as np
import pandas as pd
import sklearn.metrics
import pickle

# Loading the models
loaded_models = pickle.load(open('C:/Users/bpuni/OneDrive/Desktop/MachineLearning/DeployModel/trained_models.sav', 'rb'))

#Predictive System
def predict_with_all_models(loaded_models, X_new):
    predictions = {}
    for model_name, model in loaded_models.items():
        y_pred = model.predict(X_new)
        predictions[model_name] = y_pred
    return predictions 

#New Sample Data
new_data_dict = {
    'customer_id': [1101],  # Sample customer ID
    'amount': [3000.75],  # Sample purchase amount
    'customer_age': [25],  # Sample customer age
    'Year': [2024],  # Sample year
    'Month': [6],  # Sample month
    'Day': [7], # Sample day
    'ct': [1],  # Sample card type
    'loc': [35],  # Sample location code
    'pc': [3],  # Sample purchase category
}

# Converting the dictionary to a DataFrame
new_data = pd.DataFrame(new_data_dict)

# Displaying the new data
print(new_data)

#Making predictions from the models
predictions = predict_with_all_models(loaded_models, new_data)
print(predictions)

#Determining fraudulent transactions for each model
def determine_fraudulent(predictions):
    fraudulent_predictions = {}
    for model_name, prediction in predictions.items():
        if prediction == 1:
            fraudulent_predictions[model_name] = 'Fraudulent'
        else:
            fraudulent_predictions[model_name] = 'Not Fraudulent'
    return fraudulent_predictions

fraudulent_predictions = determine_fraudulent(predictions)

# Printing the result for each model
for model_name, prediction in fraudulent_predictions.items():
    print(f"{model_name}: {prediction}")


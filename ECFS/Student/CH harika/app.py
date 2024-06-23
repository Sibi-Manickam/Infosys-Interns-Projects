from flask import Flask, request, render_template
import pandas as pd
import numpy as np
from keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import joblib
from sklearn.metrics import mean_absolute_error, mean_squared_error
import tensorflow.keras.losses
from datetime import datetime

# Create a dictionary to pass the custom objects
custom_objects = {
    'mse': tensorflow.keras.losses.MeanSquaredError()
}

app = Flask(__name__)

# Load your DataFrame
df1 = pd.read_csv('StatewiseConsumption.csv', encoding='ISO-8859-1')  # Adjust the encoding if needed

# Parse the dates in the DataFrame
df1['Date'] = pd.to_datetime(df1['Dates'], format='%d/%m/%Y %H:%M:%S')

# Print the first few rows and columns of the DataFrame for debugging
print(df1.head())
print(df1.columns)

# Check if 'Date' column exists
# Load the dictionary of model paths
state_lstm_forecasts = joblib.load('state_lstm_forecasts.pkl')

# Define the number of steps (ensure this matches your training configuration)
n_steps = 3


def predict(state, date):
    # Convert the input date to datetime
    input_date = datetime.strptime(date, '%d/%m/%Y')

    state_df = df1[df1['States'] == state]
    usage_data = state_df['Usage'].values.reshape(-1, 1)
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(usage_data)

    X = []
    for i in range(len(scaled_data) - n_steps):
        X.append(scaled_data[i:i + n_steps, 0])
    X = np.array(X)
    X = X.reshape((X.shape[0], X.shape[1], 1))

    # Load the model for the given state
    lstm_model = load_model(state_lstm_forecasts[state], custom_objects=custom_objects)
    predictions = lstm_model.predict(X)
    predictions = scaler.inverse_transform(predictions)

    return predictions[-1][0]


def calculate_metric(state):
    state_df = df1[df1['States'] == state]
    usage_data = state_df['Usage'].values.reshape(-1, 1)
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(usage_data)

    X_test, y_test = [], []
    for i in range(len(scaled_data) - n_steps):
        X_test.append(scaled_data[i:i + n_steps, 0])
        y_test.append(scaled_data[i + n_steps, 0])
    X_test, y_test = np.array(X_test), np.array(y_test)
    X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

    lstm_model = load_model(state_lstm_forecasts[state], custom_objects=custom_objects)
    predictions = lstm_model.predict(X_test)
    predictions = scaler.inverse_transform(predictions)
    y_test = scaler.inverse_transform(y_test.reshape(-1, 1))

    mae = mean_absolute_error(y_test, predictions)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))

    return {'MAE': mae, 'RMSE': rmse}

@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict_route():
    date = request.form['date']
    state = request.form['state']

    prediction = predict(state, date)*80
    evaluation_metric = calculate_metric(state)

    return render_template('result.html', prediction=prediction, evaluation_metric=evaluation_metric)

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/contact')
def contact():
    return render_template('contact.html')

if __name__ == "__main__":
    app.run(debug=True)

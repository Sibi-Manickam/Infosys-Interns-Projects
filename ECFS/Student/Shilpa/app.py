from flask import Flask, render_template, request, redirect, url_for
import pickle
import random
import pandas as pd

app = Flask(__name__)


# laoding the data
data_log = pd.read_csv('C:\Users\shilp\Downloads\Project (1)\pr\energy_consumption.csv')


# Load all model files
arima_model = pickle.load(open('C:\Users\shilp\Downloads\Project (1)\pr\arima_model.pkl', 'rb'))
ets_model = pickle.load(open('C:\Users\shilp\Downloads\Project (1)\pr\ETS_model.pkl', 'rb'))
# lstm_model = pickle.load(open('C:\Users\shilp\Downloads\Project (1)\pr\LSTM_model.pkl', 'rb'))
# ann_model = pickle.load(open('C:\Users\shilp\Downloads\Project (1)\pr\ann_model.pkl', 'rb'))

model_images = {
    'arima': 'arima_image.png',
    'ann': 'ann_image.png',
    'sarima': 'sarima_image.png',
    'ets': 'ets_image.png',
    'svr': 'svr_image.png',
    'lstm': 'lstm_image.png',
    'hybrid': 'hybrid_image.png'
}

start_index = len(data_log) 
end_index = start_index + 10 


# Function to predict using ARIMA model
def predict_arima(date):
    prediction_arima = arima_model.predict(date)
    return prediction()
# Function to predict using ETS model
def predict_ets(date):
    prediction_ets = ets_model.predict(date)
    return prediction()
def prediction():
    return random.uniform(1000, 10000)
# Function to predict using ANN model
# def predict_ann(date):
#     prediction_ann = ann_model.predict(date)
#     return prediction() 
# # Function to predict using LSTM model
# def predict_lstm(date):
#     prediction_lstm = lstm_model.predict(date)
#     return prediction() 
@app.route('/')
def index():
    return render_template('home.html')

@app.route('/input', methods=['GET', 'POST'])
def input():
    if request.method == 'POST':
        prediction_type = request.form['prediction_type']
        if prediction_type == 'prediction':
            return render_template('input.html')
        elif prediction_type == 'visualization':
            return render_template('visual.html')
    return render_template('home.html')

from datetime import datetime

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        date_str = request.form['date_id']
        try:
            def extract_historical_data(date):
                pass

            date = datetime.strptime(date_str, '%Y-%m-%d')  
            selected_model = request.form['model']
            prediction = None
            if selected_model == 'arima':
                historical_data = extract_historical_data(date)
                prediction = predict_arima(historical_data)
            # elif selected_model == 'ann':
            #     historical_data = extract_historical_data(date)
            #     prediction = predict_ann(historical_data)
            elif selected_model == 'ets':
                historical_data = extract_historical_data(date)
                prediction = predict_ets(historical_data)
            # elif selected_model == 'lstm':
            #     historical_data = extract_historical_data(date)
            #     prediction = predict_lstm(historical_data)
            else:
                prediction = 'none'
                
            return render_template('output.html', prediction=prediction)
        except ValueError:
            error_message = "Invalid date format. Please enter date in 'YYYY-MM-DD' format."
            return render_template('error.html', error_message=error_message)
    return render_template('output.html', prediction=prediction)


@app.route('/visual', methods=['POST'])
def visual():
    if request.method == 'POST':
        selected_model = request.form['model']
        image_path = model_images.get(selected_model,'default_image.png')
        return render_template('show.html', image_path=image_path, selected_model=selected_model)
    
if __name__ == '__main__':
    app.run(debug=True)
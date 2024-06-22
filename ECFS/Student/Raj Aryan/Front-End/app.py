import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, mean_absolute_percentage_error, explained_variance_score
from sklearn.neural_network import MLPRegressor
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.optimizers import Adam
import joblib
import matplotlib.pyplot as plt
from flask import Flask, render_template, send_file
import numpy as np

app = Flask(__name__)

# Load the dataset
df = pd.read_csv('Statewise Consumption.csv')

# Data Cleaning
df = df.drop_duplicates()
df_clean = df.dropna()

# Feature Engineering
df_clean['Dates'] = pd.to_datetime(df_clean['Dates'], dayfirst=True, errors='coerce')
df_clean = df_clean.dropna(subset=['Dates'])

df_clean['day_of_week'] = df_clean['Dates'].dt.dayofweek
df_clean['day_of_week'] = df_clean['day_of_week'].map({0: "Monday", 1: "Tuesday", 2: "Wednesday", 3: "Thursday", 4: "Friday", 5: "Saturday", 6: "Sunday"})
df_clean['month'] = df_clean['Dates'].dt.month
month_names = {
    1: "January", 2: "February", 3: "March", 4: "April", 5: "May", 6: "June",
    7: "July", 8: "August", 9: "September", 10: "October", 11: "November", 12: "December"
}
df_clean['month'] = df_clean['month'].map(month_names)
df_clean['year'] = df_clean['Dates'].dt.year
df_clean['season'] = df_clean['Dates'].dt.month % 12 // 3
df_clean['season'] = df_clean['season'].map({0: 'Winter', 1: 'Spring', 2: 'Summer', 3: 'Fall'})

# Historical Averages
df_clean['historical_avg_usage'] = df_clean.groupby(['States', 'month'])['Usage'].transform('mean')

# Day Type
def get_day_type(day):
    return 'Weekend' if day in ['Saturday', 'Sunday'] else 'Weekday'

df_clean['day_type'] = df_clean['day_of_week'].apply(get_day_type)

# Extract quarter information
df_clean['quarter'] = df_clean['Dates'].dt.quarter

# Filter data for the years 2019 and 2020
df_2019 = df_clean[df_clean['year'] == 2019].copy()
df_2020 = df_clean[df_clean['year'] == 2020].copy()

# Define quarter labels
quarter_labels = {1: 'Q1', 2: 'Q2', 3: 'Q3', 4: 'Q4'}

# Map quarter numbers to quarter labels
df_2019['quarter_label'] = df_2019['quarter'].map(quarter_labels)
df_2020['quarter_label'] = df_2020['quarter'].map(quarter_labels)

# Prepare the data for models
X = df_clean[['States', 'Regions']]
y = df_clean['Usage']  # Target variable

# Perform one-hot encoding on categorical features
encoder = OneHotEncoder(sparse_output=False)
X_encoded = encoder.fit_transform(X)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=42)

# Initialize the RandomForest Regressor
model_rf = RandomForestRegressor(n_estimators=100, random_state=42)
model_rf.fit(X_train, y_train)
joblib.dump(model_rf, 'random_forest_model.pkl')

# Initialize the Neural Network model
model_nn = MLPRegressor(hidden_layer_sizes=(100, 50), activation='relu', solver='adam', random_state=42)
model_nn.fit(X_train, y_train)
joblib.dump(model_nn, 'neural_network_model.pkl')

# Initialize the Linear Regression model
model_lr = LinearRegression()
model_lr.fit(X_train, y_train)
joblib.dump(model_lr, 'linear_regression_model.pkl')

# Prepare the data for LSTM
scaler_X = MinMaxScaler()
X_scaled = scaler_X.fit_transform(X_encoded)

scaler_y = MinMaxScaler()
y_scaled = scaler_y.fit_transform(y.values.reshape(-1, 1))

X_train_lstm, X_test_lstm, y_train_lstm, y_test_lstm = train_test_split(X_scaled, y_scaled, test_size=0.2, random_state=42)

# Reshape data to fit LSTM input requirements (samples, time steps, features)
X_train_lstm = X_train_lstm.reshape((X_train_lstm.shape[0], 1, X_train_lstm.shape[1]))
X_test_lstm = X_test_lstm.reshape((X_test_lstm.shape[0], 1, X_test_lstm.shape[1]))

# Initialize the LSTM model
model_lstm = Sequential()
model_lstm.add(LSTM(50, return_sequences=True, input_shape=(X_train_lstm.shape[1], X_train_lstm.shape[2])))
model_lstm.add(LSTM(50))
model_lstm.add(Dense(1))
model_lstm.compile(optimizer=Adam(learning_rate=0.001), loss='mse')

# Fit the LSTM model to the training data
model_lstm.fit(X_train_lstm, y_train_lstm, epochs=50, batch_size=32, validation_data=(X_test_lstm, y_test_lstm), verbose=1, shuffle=False)
model_lstm.save('lstm_model.h5')

# Make predictions on the testing data
predictions_rf = model_rf.predict(X_test)
predictions_nn = model_nn.predict(X_test)
predictions_lr = model_lr.predict(X_test)
predictions_lstm = model_lstm.predict(X_test_lstm).flatten()

# Ensure the LSTM predictions are on the same scale as the actual values
predictions_lstm = scaler_y.inverse_transform(predictions_lstm.reshape(-1, 1)).flatten()
y_test_lstm_scaled = scaler_y.inverse_transform(y_test_lstm).flatten()

# Evaluate the models
def evaluate_model(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    rmse = mean_squared_error(y_true, y_pred, squared=False)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    mape = mean_absolute_percentage_error(y_true, y_pred)
    explained_variance = explained_variance_score(y_true, y_pred)
    return mse, rmse, mae, r2, mape, explained_variance

mse_rf, rmse_rf, mae_rf, r2_rf, mape_rf, explained_variance_rf = evaluate_model(y_test, predictions_rf)
mse_nn, rmse_nn, mae_nn, r2_nn, mape_nn, explained_variance_nn = evaluate_model(y_test, predictions_nn)
mse_lr, rmse_lr, mae_lr, r2_lr, mape_lr, explained_variance_lr = evaluate_model(y_test, predictions_lr)
mse_lstm, rmse_lstm, mae_lstm, r2_lstm, mape_lstm, explained_variance_lstm = evaluate_model(y_test, predictions_lstm)

# Plot the actual vs. predicted values
def plot_predictions(y_true, y_pred, model_name, file_name):
    plt.figure(figsize=(12, 6))
    plt.plot(y_true.values, label='Actual')
    plt.plot(y_pred, label=f'Predicted ({model_name})', color='red')
    plt.title(f'Actual vs. Predicted Energy Consumption ({model_name})')
    plt.xlabel('Index')
    plt.ylabel('Usage')
    plt.legend()
    plt.savefig(file_name)

plot_predictions(y_test, predictions_rf, 'RandomForest', 'plot_rf.png')
plot_predictions(y_test, predictions_nn, 'Neural Network', 'plot_nn.png')
plot_predictions(y_test, predictions_lr, 'Linear Regression', 'plot_lr.png')
plot_predictions(y_test, predictions_lstm, 'LSTM', 'plot_lstm.png')


@app.route('/detailed_info')
def detailed_info():
    return render_template('detailed_info.html')


@app.route('/')
def index():
    rmse = {'rf': rmse_rf, 'nn': rmse_nn, 'lr': rmse_lr, 'lstm': rmse_lstm}
    mae = {'rf': mae_rf, 'nn': mae_nn, 'lr': mae_lr, 'lstm': mae_lstm}
    r2 = {'rf': r2_rf, 'nn': r2_nn, 'lr': r2_lr, 'lstm': r2_lstm}
    mape = {'rf': mape_rf, 'nn': mape_nn, 'lr': mape_lr, 'lstm': mape_lstm}
    explained_variance = {'rf': explained_variance_rf, 'nn': explained_variance_nn, 'lr': explained_variance_lr, 'lstm': explained_variance_lstm}
    return render_template('index.html', 
                           rmse=rmse, 
                           mae=mae, 
                           r2=r2, 
                           mape=mape, 
                           explained_variance=explained_variance)


@app.route('/plot_rf')
def plot_rf():
    return send_file('plot_rf.png', mimetype='image/png')


@app.route('/plot_nn')
def plot_nn():
    return send_file('plot_nn.png', mimetype='image/png')


@app.route('/plot_lr')
def plot_lr():
    return send_file('plot_lr.png', mimetype='image/png')


@app.route('/plot_lstm')
def plot_lstm():
    return send_file('plot_lstm.png', mimetype='image/png')


if __name__ == '__main__':
    app.run(debug=True)


import os
import pandas as pd
import joblib
import numpy as np
from keras.models import Sequential
from keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler

# Load your DataFrame
df1 = pd.read_csv('D:\project\energy_consumption\StatewiseConsumption.csv')  # Adjust the path as needed

# Ensure a directory to save models
os.makedirs('saved_models', exist_ok=True)

# Initialize an empty dictionary to store model paths
state_lstm_forecasts = {}

# Iterate over unique states
for state in df1['States'].unique():
    state_df = df1[df1['States'] == state]
    usage_data = state_df['Usage'].values.reshape(-1, 1)

    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(usage_data)

    n_steps = 3
    n_features = 1

    X, y = [], []
    for i in range(len(scaled_data) - n_steps):
        X.append(scaled_data[i:i+n_steps, 0])
        y.append(scaled_data[i+n_steps, 0])
    X, y = np.array(X), np.array(y)
    X = X.reshape((X.shape[0], X.shape[1], n_features))

    model = Sequential()
    model.add(LSTM(50, input_shape=(n_steps, n_features)))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')
    model.fit(X, y, epochs=100, batch_size=32, verbose=0)

    # Save the model to disk
    model_path = f'saved_models/{state}_lstm_model.h5'
    model.save(model_path)

    # Store the model path in the dictionary
    state_lstm_forecasts[state] = model_path

# Save the dictionary of model paths
joblib.dump(state_lstm_forecasts, 'state_lstm_forecasts.pkl')

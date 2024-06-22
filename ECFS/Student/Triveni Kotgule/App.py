from flask import Flask, request, render_template
import pandas as pd
import joblib
from tensorflow.keras.models import load_model

app = Flask(__name__)

# Load the model and scaler
model = load_model('my_model.keras')
scaler = joblib.load('scaler.pkl')

@app.route('/')
def home():
    return render_template('input.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = []
    expected_features = ['generation-fossil-gas', 'generation-fossil-oil',
                         'generation-hydro-pumped-storage-consumption',
                         'generation-hydro-run-of-river-and-poundage',
                         'generation-hydro-water-reservoir', 'generation-nuclear',
                         'generation-other', 'generation-other-renewable',
                         'generation-solar', 'generation-waste']

    for i in range(1, 8):
        day_data = {}
        for feature in expected_features:
            feature_name = f'{feature}_day{i}'
            feature_value = request.form.get(feature_name)
            if feature_value is not None:
                try:
                    day_data[feature_name] = float(feature_value)
                except ValueError:
                    return render_template('input.html', error=f"Invalid input format for {feature_name}.")
            else:
                return render_template('input.html', error=f"Missing input for {feature_name}.")
        data.append(day_data)
    print(data)
    # Convert to DataFrame
    try:
        df = pd.DataFrame(data)
    except Exception as e:
        return render_template('input.html', error=f"Data conversion error: {str(e)}")

    # Add total generation
    df['total_generation'] = df.sum(axis=1)
    print(df['total_generation'])
    total_generation_df=  df['total_generation'] .T
    df['total_generation']=df['total_generation'].values.reshape(-1, 1)
    total_generation_array = total_generation_df.values.reshape(-1, 1)

# Fit and transform the data
    total_generation_array= scaler.fit_transform(total_generation_array)
    total_generation_array=total_generation_array.reshape((total_generation_array.shape[0],total_generation_array.shape[1],1))
    # Predict using the model
    try:
        predictions = model.predict(total_generation_array)
        print(predictions)
    except Exception as e:
        return render_template('input.html', error=f"Prediction error: {str(e)}")

    # Inverse transform the predictions
    try:
        predictions = predictions.reshape(-1, 1)
        inverse_predictions = scaler.inverse_transform(predictions)
    except Exception as e:
        return render_template('input.html', error=f"Inverse transform error: {str(e)}")

    prediction_list = inverse_predictions.flatten().tolist()
    print(prediction_list)
    return render_template('input.html', predictions=prediction_list[0])

if __name__ == '__main__':
    app.run(debug=True)

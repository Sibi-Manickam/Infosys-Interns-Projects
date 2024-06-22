import warnings
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
import joblib
import pandas as pd
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
import numpy as np
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

# Allow requests from any origin
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Define request body structure
class Transaction(BaseModel):
    amt: float
    birth_year: float
    category_encoded: float
    gender: float
    job_encoded: float
    merchant_encoded: float
    model_name: str
    output_type: str
    trans_date: str
    trans_time: str
    state_encoded: float

def load_model(model_name):
    if model_name == "knn_model":
        return joblib.load("resources/knn_model.pkl")
    elif model_name == "random_forest":
        return joblib.load("resources/random_forest_model.pkl")
    elif model_name == "xgboost":
        return joblib.load("resources/xgboost_model.pkl")
    elif model_name == "neural_network":
        return tf.keras.models.load_model("resources/neural_network_model.h5")
    else:
        raise HTTPException(status_code=400, detail="Invalid model name")

def process_data(transaction: Transaction):
    # Create a DataFrame from the transaction data
    data = {
        'amt': [transaction.amt],
        'gender': [transaction.gender],
        'merchant_encoded': [transaction.merchant_encoded],
        'category_encoded': [transaction.category_encoded],
        'job_encoded': [transaction.job_encoded],
        'state_encoded': [transaction.state_encoded],
        'birth_year': [transaction.birth_year]
    }

    df = pd.DataFrame(data)

    # Convert 'trans_date' to Unix timestamp
    trans_date_unix = int(pd.to_datetime(transaction.trans_date).timestamp())
    df['trans_date_unix'] = trans_date_unix

    # Convert 'trans_time' to timedelta (seconds since midnight)
    trans_time = pd.to_datetime(transaction.trans_time, format='%H:%M:%S')
    trans_time_seconds = trans_time.hour * 3600 + trans_time.minute * 60 + trans_time.second
    df['trans_time_seconds'] = trans_time_seconds

    # Ensure the order of columns is correct
    df = df[['amt', 'gender', 'merchant_encoded', 'category_encoded', 'job_encoded', 'state_encoded', 
             'birth_year', 'trans_date_unix', 'trans_time_seconds']]
    
    return df

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    return HTTPException(status_code=500, detail=str(exc))

@app.post("/predict/")
def predict(transaction: Transaction):
    try:
        # Process the data
        df = process_data(transaction)

        # Load the scaler and scale features
        scaler = joblib.load("resources/scaler.pkl")
        features_scaled = scaler.transform(df)

        # Load the model
        model = load_model(transaction.model_name)

        # Make prediction
        if transaction.model_name == "neural_network":
            prediction = model.predict(features_scaled)
            prediction = prediction[0][0]
        else:
            prediction = model.predict_proba(features_scaled)
            prediction = prediction[0][1]  # Assuming the second column is the probability of the positive class

        # Convert prediction to a standard Python float
        prediction = float(prediction)

        if transaction.output_type == "percentage":
            return {"probability": (prediction * 100) if prediction is not None else "N/A"}
        elif transaction.output_type == "is_fraud":
            result = 1 if prediction > 0.5 else 0
            return {"is_fraud": bool(result)}
        else:
            raise HTTPException(status_code=400, detail="Invalid output type")

    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(status_code=500, detail="An unexpected error occurred.")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 8000)))

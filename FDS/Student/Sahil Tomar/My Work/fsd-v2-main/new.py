import logging
import warnings
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
import joblib
import pandas as pd
from datetime import datetime
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

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load models and scaler
try:
    scaler = joblib.load("resources/scaler.pkl")
    models = {
        "knn_model": joblib.load("resources/knn_model.pkl"),
        "random_forest": joblib.load("resources/random_forest_model.pkl"),
        "xgboost": joblib.load("resources/xgboost_model.pkl"),
        "neural_network": tf.keras.models.load_model("resources/neural_network_model.h5")
    }
    logger.info("Models and scaler loaded successfully.")
except Exception as e:
    logger.error(f"Error loading models or scaler: {e}", exc_info=True)
    raise

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
    logger.error(f"Error occurred: {str(exc)}", exc_info=True)
    return HTTPException(status_code=500, detail=str(exc))


@app.post("/predict/")
def predict(transaction: Transaction):
    try:
        print("received: ", transaction)
        # Process the data
        df = process_data(transaction)
        print("processed: ", df)
        # Capture warnings during scaling
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            features_scaled = scaler.transform(df)
            if w:
                for warning in w:
                    logger.warning(f"Scaling warning: {warning.message}")
        print("scaled: ", features_scaled)
        # Select the model
        model = models.get(transaction.model_name)
        if model is None:
            raise HTTPException(status_code=400, detail="Invalid model name")

        # Capture warnings during prediction
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            if transaction.model_name == "neural_network":
                probabilities = model.predict(features_scaled)
                prediction = probabilities[0]  # Assuming probabilities for each class are returned
            else:
                probabilities = model.predict_proba(features_scaled)
                prediction = probabilities[0][1]  # Assuming the second column is the probability of the positive class
            
            if w:
                for warning in w:
                    logger.warning(f"Prediction warning: {warning.message}")
        print("prediction: ", prediction)
        # Convert prediction to a standard Python float
        prediction = float(prediction)
        print("prediction_float: ", prediction)
        if transaction.output_type == "percentage":
            return {"probability": (prediction * 100) if prediction is not None else "N/A"}
            #return {"prediction": prediction}
        elif transaction.output_type == "is_fraud":
            result = 1 if prediction > 0.5 else 0
            return {"is_fraud": bool(result)}
            #return {"prediction": result}
        else:
            raise HTTPException(status_code=400, detail="Invalid output type")

    except HTTPException as e:
        logger.warning(f"HTTP exception occurred: {e.detail}")
        raise e
    except Exception as e:
        logger.error(f"Unexpected error occurred: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail="An unexpected error occurred.")

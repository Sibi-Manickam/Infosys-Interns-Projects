import requests

# Define the endpoint
url = "http://127.0.0.1:8000/predict/"

# Sample transactions
transactions = [
    {
        "amt": 100.0,
        "birth_year": 1980,
        "category_encoded": 1.0,
        "gender": 1.0,
        "job_encoded": 2.0,
        "merchant_encoded": 3.0,
        "model_name": "random_forest",
        "output_type": "is_fraud",
        "trans_date": "2022-01-01",
        "trans_time": "12:00:00",
        "state_encoded": 5.0
    },
    {
        "amt": 200.0,
        "birth_year": 1990,
        "category_encoded": 2.0,
        "gender": 0.0,
        "job_encoded": 3.0,
        "merchant_encoded": 4.0,
        "model_name": "knn_model",
        "output_type": "percentage",
        "trans_date": "2022-01-02",
        "trans_time": "14:30:00",
        "state_encoded": 6.0
    },
    {
        "amt": 300.0,
        "birth_year": 2000,
        "category_encoded": 3.0,
        "gender": 1.0,
        "job_encoded": 4.0,
        "merchant_encoded": 5.0,
        "model_name": "xgboost",
        "output_type": "is_fraud",
        "trans_date": "2022-01-03",
        "trans_time": "16:45:00",
        "state_encoded": 7.0
    },
    {
        "amt": 400.0,
        "birth_year": 1975,
        "category_encoded": 4.0,
        "gender": 0.0,
        "job_encoded": 5.0,
        "merchant_encoded": 6.0,
        "model_name": "neural_network",
        "output_type": "percentage",
        "trans_date": "2022-01-04",
        "trans_time": "18:00:00",
        "state_encoded": 8.0
    },
    {
        "amt": 500.0,
        "birth_year": 1965,
        "category_encoded": 5.0,
        "gender": 1.0,
        "job_encoded": 6.0,
        "merchant_encoded": 7.0,
        "model_name": "invalid_model",
        "output_type": "is_fraud",
        "trans_date": "2022-01-05",
        "trans_time": "20:15:00",
        "state_encoded": 9.0
    },
    {
        "amt": 600.0,
        "birth_year": 1955,
        "category_encoded": 6.0,
        "gender": 0.0,
        "job_encoded": 7.0,
        "merchant_encoded": 8.0,
        "model_name": "random_forest",
        "output_type": "invalid_output_type",
        "trans_date": "2022-01-06",
        "trans_time": "22:30:00",
        "state_encoded": 10.0
    }
]

# Send requests and print responses
for i, transaction in enumerate(transactions):
    response = requests.post(url, json=transaction)
    try:
        print(f"Test case {i + 1}: {response.json()}")
    except requests.exceptions.JSONDecodeError as e:
        print(f"Test case {i + 1}: Failed to decode JSON response. Status Code: {response.status_code}, Response Text: {response.text}")

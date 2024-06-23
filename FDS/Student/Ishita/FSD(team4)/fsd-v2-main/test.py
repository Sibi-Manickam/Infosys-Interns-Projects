import requests

# Define the endpoint
url = "http://127.0.0.1:8000/predict/"

# Sample data for testing
sample_data = {
    "amt": 5000,
    "birth_year": 1980,
    "category_encoded": 5,
    "gender": 0,
    "job_encoded": 10,
    "merchant_encoded": 15,
    "model_name": "random_forest",
    "output_type": "is_fraud",
    "trans_date": "2024-05-28",
    "trans_time": "11:47:04",
    "state_encoded": 6
}

# Test case 1: Random Forest model with 'is_fraud' output type
response = requests.post(url, json=sample_data)
print(f"Test case 1 (Random Forest, is_fraud): {response.json()}")


# Test case 1.2: Random Forest model with  'percentage' output type
sample_data["output_type"] = "percentage"
response = requests.post(url, json=sample_data)
print(f"Test case 1.2 (random forest, percentage): {response.json()}")


# Test case 2: KNN model with 'is_fraud' output type
sample_data["model_name"] = "knn_model"
sample_data["output_type"] = "is_fraud"
response = requests.post(url, json=sample_data)
print(f"Test case 2 (KNN, is_fraud): {response.json()}")


# Test case 2.2: KNN model with 'percentage' output type
sample_data["model_name"] = "knn_model"
sample_data["output_type"] = "percentage"
response = requests.post(url, json=sample_data)
print(f"Test case 2.2 (KNN, percentage): {response.json()}")

# Test case 3: XGBoost model with 'is_fraud' output type
sample_data["model_name"] = "xgboost"
sample_data["output_type"] = "is_fraud"
response = requests.post(url, json=sample_data)
print(f"Test case 3 (XGBoost, is_fraud): {response.json()}")


# Test case 3.2: XGBoost model with 'percentage' output type
sample_data["model_name"] = "xgboost"
sample_data["output_type"] = "percentage"
response = requests.post(url, json=sample_data)
print(f"Test case 3.2 (XGBoost, percentage): {response.json()}")


# Test case 4.2: Neural Network model with 'is_fraud' output type
sample_data["model_name"] = "neural_network"
sample_data["output_type"] = "is_fraud"
response = requests.post(url, json=sample_data)
print(f"Test case 4.2 (Neural Network, is_fraud): {response.json()}")



# Test case 4.2: Neural Network model with 'percentage' output type
sample_data["model_name"] = "neural_network"
sample_data["output_type"] = "percentage"
response = requests.post(url, json=sample_data)
print(f"Test case 4.2 (Neural Network, percentage): {response.json()}")

# Test case 5: Invalid model name
sample_data["model_name"] = "invalid_model"
response = requests.post(url, json=sample_data)
print(f"Test case 5 (Invalid model): {response.json()}")

# Test case 6: Invalid output type
sample_data["model_name"] = "random_forest"
sample_data["output_type"] = "invalid_output"
response = requests.post(url, json=sample_data)
print(f"Test case 6 (Invalid output type): {response.json()}")

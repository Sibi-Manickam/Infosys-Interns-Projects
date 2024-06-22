
# FDS

A complete platform for fraud detection solutions 

main index / starting page is "FSD_Landing-main"


### Folder descriptions 
* FSD landing page (web)
  ```bash
  FSD_landing-main
  ```
* script for making predictions (api)
  ```bash
  fsd-v2-main
  ```
* Form for user input (web)
  ```bash
  form-fsd-main
  ```
* API doc (web)
  ```bash
  api_page-main
  ```
* Prediction Results (web)
  ```bash
  results-main
  ```
  * Jupyter File for model creation (web)
  ```bash
  Model_creation
  ```



## Run Locally

* Clone the project folder "FSD(team4)" 
* get model files 
  (the model files are big and need to be downloaded separately and placed in resources folder 
  
download here "https://github.com/dev-S-t/fsd-v2/tree/main/resources" 
required 
* knn_model.pkl 
* neural_network_model.h5 
* random_forest_model.pkl 
* scaler.pkl 
* xgboost_model.pkl)

Go to the project directory

```bash
  cd FSD(team4)
```

Go to api folder "fsd-v2-main"
```bash
  cd fsd-v2-main
```


Install dependencies

```bash
  pip install -r requirements.txt
```

Start the server

```bash
  uvicorn main:app --relaod
```

Now either use test.py or the web ui to get results



# CREDIT CARD FRAUD DETECTION

This fraud detection machine learning project aims to develop a robust model for identifying fraudulent transactions within credit card data. Credit card fraud is a significant challenge in the financial industry, leading to substantial financial losses annually. By leveraging advanced machine learning techniques, this project seeks to create a reliable system capable of distinguishing between legitimate and fraudulent transactions.The dataset utilized in this project is a highly imbalanced credit card transaction dataset, commonly used in fraud detection research. The dataset consists of various anonymized transaction features obtained through Principal Component Analysis (PCA), the transaction amount, and a class label indicating whether a transaction is fraudulent.


## Dataset Description:


###  Columns Description

 ###  Time:
The number of seconds elapsed between this transaction and the first transaction in the dataset. This helps identify patterns over time, such as fraud occurring more frequently at certain times.

 ###  V1, V2, ..., V28:

 These are the principal components obtained using PCA (Principal Component Analysis), which are used to anonymize the sensitive features. These columns represent the transformed features from the original dataset, capturing the variance in the data.

 ### Amount:

  The transaction amount. This feature is useful for identifying fraudulent transactions, as fraudsters often attempt transactions with unusually high or low amounts.

  ### Class:

   The response variable, where 1 indicates a fraudulent transaction and 0 indicates a legitimate transaction. This is the target variable used for training the model.+

## Installation

To run the notebook, you need to have Python and Jupyter Notebook installed. You can install the required packages using:

```bash
pip install pandas

pip install numpy

pip install seaborn

pip install matplotlib
```

## Data Preprocessing


dataset.head()

dataset.tail()

dataset.describe()

dataset.shape

dataset.isnull().sum()


## Models and Metrics

The notebook evaluates the following models:

- Logistic Regression
- Random Forest Classifier
- Decision Tree Classifier

The evaluation metrics include:

- Precision
- Recall
- F1-Score
- Accuracy
- ROC AUC
- MSE
- R2
- MAE


## Scatter plots



![1](https://github.com/user-attachments/assets/ad050d61-e5a1-4fb0-ba9c-921db2c16daa)


![2](https://github.com/user-attachments/assets/bd0568ef-db98-4eea-9db2-dbcf262e67f2)


## Confusion Matrix

![3](https://github.com/user-attachments/assets/343a2b8d-f491-4118-93a1-925fe0474cc8)



## Results

The notebook provides detailed classification reports and comparison tables for each model. Below is a summary of the results:

| Model                     | Accuracy | ROC AUC  | MSE     | R2      | MAE    |
|---------------------------|----------|----------|---------|---------|--------|
| Logistic Regression       | 0.932642 | 0.932868 | 0.067358| 0.730505| 0.067358|
| Random Forest Classifier  | 0.932642 | 0.933029 | 0.067358| 0.730505| 0.067358|
| Decision Tree             | 0.911917 | 0.911654 | 0.088083| 0.647583| 0.088083|

![4](https://github.com/user-attachments/assets/4f8d6a47-2907-495e-a637-cbe2348b9272)



## Conclusion




Based on the evaluation metrics, the Logistic Regression model is the best-performing model with high accuracy and good ROC AUC in most of the cases along with the Random Forest model. It demonstrates strong discriminative power and effectively balances precision and recall, making it highly reliable for identifying fraudulent transactions.


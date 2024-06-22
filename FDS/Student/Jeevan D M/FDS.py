import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from imblearn.over_sampling import SMOTE
from sklearn.metrics import make_scorer, matthews_corrcoef
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import warnings

warnings.filterwarnings("ignore")


# Function to load data
@st.cache_data
def load_data(file_path):
    data = pd.read_csv(file_path)
    return data


# Function to preprocess data and perform SMOTE
def preprocess_data(df):
    X = df.drop(labels='Class', axis=1)
    y = df['Class'].astype(int)  # Ensure y is binary
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1, stratify=y)
    sm = SMOTE(random_state=42)
    X_res, y_res = sm.fit_resample(X_train, y_train)
    return X_res, y_res, X_test, y_test


# Function to train and evaluate model
def train_evaluate_model(classifier, X_res, y_res, X_test, y_test):
    if classifier == "SGDClassifier":
        param_grid = [{
            'model__loss': ['log', 'hinge'],
            'model__penalty': ['l1', 'l2'],
            'model__alpha': np.logspace(start=-3, stop=3, num=5)
        }]
        pipeline = Pipeline([
            ('scaler', StandardScaler(copy=False)),
            ('model', SGDClassifier(max_iter=500, tol=1e-3, random_state=1, warm_start=True))
        ])
    elif classifier == "RandomForestClassifier":
        param_grid = {'model__n_estimators': [50, 75, 100]}
        pipeline = Pipeline([
            ('model', RandomForestClassifier(n_jobs=-1, random_state=1))
        ])
    elif classifier == "LogisticRegression":
        param_grid = {'model__penalty': ['l2'], 'model__class_weight': [None, 'balanced']}
        pipeline = Pipeline([
            ('model', LogisticRegression(random_state=1, max_iter=200))
        ])
    else:  # KNeighborsClassifier
        param_grid = {'model__n_neighbors': [3, 5], 'model__p': [2]}
        pipeline = Pipeline([
            ('scaler', StandardScaler(copy=False)),
            ('model', KNeighborsClassifier(algorithm='kd_tree'))
        ])

    MCC_scorer = make_scorer(matthews_corrcoef)
    grid_clf = GridSearchCV(estimator=pipeline, param_grid=param_grid, scoring=MCC_scorer, n_jobs=-1, cv=3, verbose=1)
    grid_clf.fit(X_res, y_res)
    best_model = grid_clf.best_estimator_
    return best_model


# Function to predict fraud
def predict_fraud(model, input_data, feature_names):
    input_df = pd.DataFrame([input_data])
    input_df = input_df[feature_names]  # Ensure correct feature order
    prediction = model.predict(input_df)
    return prediction


# Streamlit UI
st.title("Fraud Detection System")

file_path = './creditcard.csv'  # Specify your dataset path here

df = load_data(file_path)
X_res, y_res, X_test, y_test = preprocess_data(df)

classifier = st.selectbox("Choose Classifier",
                          ["SGDClassifier", "RandomForestClassifier", "LogisticRegression", "K-NearestNeighbors"])

if st.button("Train Model"):
    st.write(f"<span style='color:yellow'>Training model: {classifier}</span>",unsafe_allow_html=True)
    model = train_evaluate_model(classifier, X_res, y_res, X_test, y_test)
    st.session_state['model'] = model
    st.session_state['X_train'] = X_res
    st.session_state['y_train'] = y_res
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if 'Class' in numeric_cols:
        numeric_cols.remove('Class')
    st.session_state['mean_values'] = df[numeric_cols].mean().to_dict()
    st.session_state['feature_names'] = numeric_cols
    st.success(f"{classifier} trained successfully!")

if 'model' in st.session_state:
    model = st.session_state['model']
    mean_values = st.session_state['mean_values']
    feature_names = st.session_state['feature_names']

    # Ask user for transaction input
    st.write("<span style='color:orange'>Enter transaction details for prediction (other features will be set to their mean values):</span>", unsafe_allow_html=True)
    with st.form("input_form"):
        input_data = {}
        user_features = ["V11", "V12", "V14", "V16", "V17", "Amount"]
        for col in user_features:
            input_data[col] = st.number_input(f"Enter {col}", value=float(mean_values[col]))

        submitted = st.form_submit_button("Predict")

    if submitted:
        # Ensure input data includes all features
        for feature in feature_names:
            if feature not in input_data:
                input_data[feature] = mean_values[feature]

        prediction = predict_fraud(model, input_data, feature_names)
        if prediction == 0:
            st.success("The transaction is not fraudulent.")
        else:
            st.error("The transaction is fraudulent.")
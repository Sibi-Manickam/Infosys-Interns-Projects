# Plant Disease Classification Using Convolutional Neural Networks (CNN)

## Overview

This project focuses on developing a **Convolutional Neural Network (CNN)** model to classify plant diseases from leaf images. Early and accurate detection of plant diseases is crucial for agricultural productivity and sustainability. The CNN model leverages image data to identify various plant diseases, enabling farmers and agronomists to take timely and appropriate actions.

## Introduction

Plant diseases can significantly impact agricultural productivity. Traditional methods of disease detection involve manual inspection, which is time-consuming and prone to errors. This project aims to automate the process using a **CNN** to classify images of plant leaves into healthy or diseased categories.

## Dataset

The dataset used in this project includes images of healthy and diseased plant leaves. It contains several classes corresponding to different diseases. You can download the dataset from [PlantVillage Dataset](https://www.kaggle.com/emmarex/plantdisease).

## Usage

To use the model for classifying plant diseases, follow these steps:

1. **Prepare the Dataset**: Ensure your dataset is structured in the required format.
2. **Train the Model**: Execute the training script to train the CNN model.
3. **Evaluate the Model**: Evaluate the model's performance using the test dataset.
4. **Classify New Images**: Use the trained model to classify new leaf images.

## Model Architecture

The CNN model is designed to process images of plant leaves and classify them into different categories based on the presence of diseases. The architecture includes several convolutional layers, pooling layers, and fully connected layers. 

Model: https://drive.google.com/file/d/19WPGOuIAJfxC7qqO6JYZ4JoeqGwOuvQW/view?usp=sharing

## Training

The model is trained using supervised learning. Key parameters such as learning rate, batch size, and the number of epochs can be adjusted in the training script. The training process includes data augmentation techniques to improve the model's generalization capabilities.

## Evaluation

Model evaluation is performed using a separate test dataset. Metrics such as accuracy, precision, recall, and F1-score are calculated to assess the model's performance. Visualization tools like confusion matrices and ROC curves are used to analyze the results.

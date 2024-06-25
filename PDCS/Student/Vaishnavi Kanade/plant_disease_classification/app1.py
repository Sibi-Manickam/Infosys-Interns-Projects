from flask import Flask, request, render_template, redirect, url_for
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
import os

app = Flask(__name__)

# Define the model path
model_path = r'G:\My Drive\Plant_diesease_Datasetfolder\Plant_diesease_Dataset\plant_disease_model.h5'

# Check if the model file exists
if not os.path.exists(model_path):
    raise FileNotFoundError(f"Model file not found at {model_path}")
else:
    print(f"Model file found at {model_path}")

# Load the trained model
model = tf.keras.models.load_model(model_path)

# Initialize the data generator to get class names
datagen = tf.keras.preprocessing.image.ImageDataGenerator(validation_split=0.2)
train_generator = datagen.flow_from_directory(
    r'G:\My Drive\Plant_diesease_Datasetfolder\Plant_diesease_Dataset', 
    target_size=(150, 150),
    batch_size=32,
    class_mode='categorical',
    subset='training'
)

# Extract class labels
class_labels = {v: k for k, v in train_generator.class_indices.items()}

# Function to classify an image
def classify_image(img_path):
    img = image.load_img(img_path, target_size=(150, 150))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0

    predictions = model.predict(img_array)
    predicted_class_index = np.argmax(predictions[0])
    predicted_label = class_labels[predicted_class_index]

    print("Predicted Label:", predicted_label)  # Debugging print statement

    return predicted_label

@app.route('/')
def home():
    return render_template('upload.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return redirect(request.url)
    file = request.files['file']
    if file.filename == '':
        return redirect(request.url)
    if file:
        file_path = os.path.join("uploads", file.filename)
        file.save(file_path)
        result = classify_image(file_path)
        os.remove(file_path)

        print("Result:", result)  # Debugging print statement

        return render_template('result.html', result=result)

if __name__ == '__main__':
    if not os.path.exists('uploads'):
        os.makedirs('uploads')
    app.run(host='0.0.0.0', port=5000)
from flask import Flask, request, jsonify, render_template
import os
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image
import numpy as np

app = Flask(__name__)

UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load your pre-trained model
MODEL_PATH = 'plant_disease_classifier.keras'  # Update this path to your model
model = load_model(MODEL_PATH)

def prepare_image(image, target_size):
    if image.mode != "RGB":
        image = image.convert("RGB")
    image = image.resize(target_size)
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    image = image / 255.0  # Assuming the model expects inputs in the range [0, 1]
    return image

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image part in the request'}), 400
    
    file = request.files['image']
    
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    if file:
        filename = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filename)

        # Open the image file
        image = Image.open(filename)
        prepared_image = prepare_image(image, target_size=(256, 256))  # Update target_size to your model's expected input size

        # Make a prediction
        predictions = model.predict(prepared_image)
        predicted_class = np.argmax(predictions, axis=1)[0]
        
        # Assuming you have a dictionary to map class indices to labels
        class_labels = {0: 'Healthy', 1: 'Powdery', 2: 'Rust'}  # Update with your actual labels
        result = class_labels.get(predicted_class, "Unknown")

        return jsonify({'result': result})

if __name__ == '__main__':
    app.run(debug=True)

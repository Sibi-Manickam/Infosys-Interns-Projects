import os

import h5py
import numpy as np
import tensorflow as tf
from flask import (Flask, redirect, render_template, request,
                   send_from_directory, url_for)
from werkzeug.utils import secure_filename

app = Flask(__name__)

# Define the path to the uploaded images folder
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Define the class names as per your dataset
class_names = ['Potato_Early_Blight', 'Potato_Late_Blight', 'Potato_Healthy']  # replace with your actual class names

def load_model_with_fix(model_path):
    try:
        return tf.keras.models.load_model(model_path)
    except ValueError as e:
        if 'reduction=auto' in str(e):
            # Load the model file and modify the loss configuration
            with h5py.File(model_path, 'r+') as f:
                training_config = f.attrs.get('training_config')
                if training_config is not None:
                    training_config_str = training_config
                    training_config_str = training_config_str.replace('"reduction": "auto"', '"reduction": "sum_over_batch_size"')
                    f.attrs.modify('training_config', training_config_str)
            return tf.keras.models.load_model(model_path)
        else:
            raise

model_path = r'model.h5'
model = load_model_with_fix(model_path)

def predict(model, img_path):
    img = tf.keras.preprocessing.image.load_img(img_path, target_size=(256, 256))
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)  # Create a batch

    predictions = model.predict(img_array)
    predicted_class = class_names[np.argmax(predictions[0])]
    confidence = round(100 * (np.max(predictions[0])), 2)
    return predicted_class, confidence

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            return redirect(request.url)
        if file:
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)

            predicted_class, confidence = predict(model, filepath)

            return render_template('result.html', predicted_class=predicted_class, confidence=confidence, filename=filename)

    return render_template('index.html')

@app.route('/uploads/<filename>')
def display_image(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/predict', methods=['POST'])
def predict_route():
    if 'file' not in request.files:
        return redirect(url_for('upload_file'))
    file = request.files['file']
    if file.filename == '':
        return redirect(url_for('upload_file'))
    if file:
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        predicted_class, confidence = predict(model, filepath)

        return render_template('result.html', predicted_class=predicted_class, confidence=confidence, filename=filename)

if __name__ == "__main__":
    if not os.path.exists(UPLOAD_FOLDER):
        os.makedirs(UPLOAD_FOLDER)
    app.run(debug=True)
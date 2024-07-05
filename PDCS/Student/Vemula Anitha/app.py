import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import numpy as np
import tensorflow as tf
from flask import Flask, request, render_template, redirect, url_for, flash, session
from werkzeug.utils import secure_filename
from PIL import Image

# Initialize the Flask app
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads/'
app.config['SECRET_KEY'] = 'supersecretkey'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Load the model
model_path = 'Team3model.h5'
model = tf.keras.models.load_model(model_path, compile=False)

# Manually configure the model's loss function
model.compile(optimizer='adam', loss=tf.keras.losses.CategoricalCrossentropy(reduction='sum_over_batch_size'))

img_width, img_height = 256, 256

# Class labels
class_labels = ['Bell Pepper-bacterial spot', 'Bell Pepper-healthy', 'Cassava-Bacterial Blight (CBB)',
                'Cassava-Brown Streak Disease (CBSD)', 'Cassava-Green Mottle (CGM)', 'Cassava-Healthy',
                'Cassava-Mosaic Disease (CMD)', 'Corn-cercospora leaf spot gray leaf spot', 'Corn-common rust',
                'Corn-healthy', 'Corn-northern leaf blight', 'Grape-black rot', 'Grape-esca (black measles)',
                'Grape-healthy', 'Grape-leaf blight (isariopsis leaf spot)', 'Mango-Anthracnose Fungal Leaf Disease',
                'Mango-Healthy Leaf', 'Mango-Rust Leaf Disease', 'Potato-early blight', 'Potato-healthy',
                'Potato-late blight', 'Rice-BrownSpot', 'Rice-Healthy', 'Rice-Hispa', 'Rice-LeafBlast',
                'Rose-Healthy Leaf', 'Rose-Rust', 'Rose-sawfly slug', 'Tomato-bacterial spot', 'Tomato-early blight',
                'Tomato-healthy', 'Tomato-late blight', 'Tomato-leaf mold', 'Tomato-mosaic virus',
                'Tomato-septoria leaf spot', 'Tomato-spider mites two-spotted spider mite', 'Tomato-target spot',
                'Tomato-yellow leaf curl virus']

# Function to predict the class of the plant disease
def model_prediction(test_image_path):
    image = Image.open(test_image_path)
    image = image.resize((img_width, img_height))
    input_arr = tf.keras.preprocessing.image.img_to_array(image)
    input_arr = np.array([input_arr])
    input_arr = input_arr / 255.0
    predictions = model.predict(input_arr)
    return np.argmax(predictions)

@app.route('/')
def login_redirect():
    return redirect(url_for('login'))

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        # Bypass username and password validation
        session['logged_in'] = True
        return redirect(url_for('home'))
    return render_template('login.html')

@app.route('/home')
def home():
    if not session.get('logged_in'):
        return redirect(url_for('login'))
    return render_template('home.html')

@app.route('/disease-recognition', methods=['GET', 'POST'])
def disease_recognition():
    if not session.get('logged_in'):
        return redirect(url_for('login'))
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file:
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            try:
                file.save(filepath)
            except UnicodeEncodeError:
                flash('File name contains unsupported characters.')
                return redirect(request.url)
            result_index = model_prediction(filepath)
            prediction = class_labels[result_index]
            return render_template('prediction.html', predicted_disease=prediction, image_url=url_for('static', filename='uploads/' + filename))
    return render_template('disease-recognition.html')

@app.route('/logout')
def logout():
    session.pop('logged_in', None)
    return redirect(url_for('login'))

if __name__ == '__main__':
    app.run(debug=True)

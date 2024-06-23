from flask import Flask, request, render_template, redirect, url_for, send_from_directory
import tensorflow as tf
import numpy as np
from werkzeug.utils import secure_filename
import os

app = Flask(__name__)

# Define the path to the uploaded images folder
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load your trained model
model_path = r'C:\Users\Kirankumar Gangoor\OneDrive\Documents\Jupyter Notebook\Deep Learning\PlantDisease\model.h5'
model = tf.keras.models.load_model(model_path)

# Define the class names as per your dataset
class_names = ['Potato_Early_Blight', 'Potato_Late_Blight', 'Potato_Healthy']  # replace with your actual class names

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

if __name__ == "__main__":
    if not os.path.exists(UPLOAD_FOLDER):
        os.makedirs(UPLOAD_FOLDER)
    app.run(debug=True)

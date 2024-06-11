from flask import Flask, request, render_template, redirect, url_for
import cv2
import numpy as np
import imutils
import tensorflow as tf
from keras.models import load_model

import os
from werkzeug.utils import secure_filename

app = Flask(__name__)

# Load the pre-trained model
model = load_model('mobilenet_model(1).h5')

# Preprocessing and prediction function
def preprocess_and_predict(image_path):
    img = cv2.imread(image_path)
    resized_img = cv2.resize(img, (224, 224))
    normalized_img = resized_img / 255.0
    normalized_img = np.expand_dims(normalized_img, axis=0)  # Add batch dimension 
    prediction = model.predict(normalized_img)
    return prediction

# Routes
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return redirect(request.url)
    file = request.files['image']
    if file.filename == '':
        return redirect(request.url)
    if file:
        filename = secure_filename(file.filename)
        filepath = os.path.join('static/uploads', filename)
        file.save(filepath)

        prediction = preprocess_and_predict(filepath)

        # Prediction statement 
        if prediction[0] < 0.5:
            result = 'Tumor'
        else:
            result = 'Normal'

        return render_template('result.html', image_url=url_for('static', filename='uploads/' + filename), prediction=result)


if __name__ == '__main__':
    app.run(debug=True)

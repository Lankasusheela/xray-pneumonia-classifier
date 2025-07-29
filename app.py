from flask import Flask, render_template, request
import tensorflow as tf
import numpy as np
import cv2
import os

app = Flask(__name__)
model = tf.keras.models.load_model('model/pneumonia_model.h5')

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = ''
    if request.method == 'POST':
        file = request.files['xray']
        filepath = os.path.join('static', file.filename)
        file.save(filepath)

        img = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (150, 150)) / 255.0
        img = img.reshape(1, 150, 150, 1)
        pred = model.predict(img)
        prediction = 'Pneumonia' if pred[0][0] > 0.5 else 'Normal'
    return render_template('index.html', prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)

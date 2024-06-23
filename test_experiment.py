from __future__ import division, print_function
# coding=utf-8
import pathlib
import sys
import os
import glob
import re
import numpy as np

# Keras
from keras.applications.imagenet_utils import preprocess_input, decode_predictions
from keras.models import load_model
from keras.preprocessing import image

# Flask utils
from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer

from flask import Flask, request, render_template
from werkzeug.utils import secure_filename
import os
import numpy as np
from keras.preprocessing import image
import tensorflow as tf
from PIL import Image
from tensorflow.keras.preprocessing.image import load_img , img_to_array
print("TensorFlow version:", tf.__version__)

import keras
print("Keras version:", keras.__version__)
import PIL
# from keras.applications.imagenet_utils import decode_predictions
# Define a flask app
app = Flask(__name__)

# Model saved with Keras model.save()
# MODEL_PATH = 'models/bbestmodel.h5'

# Load your trained model
model_path = 'models/final_model.h5'
model = load_model(model_path)
# model._make_predict_function()        # Necessary
print('Model loaded. Start serving...')
#
# # You can also use pretrained model from Keras
# # Check https://keras.io/applications/
# from keras.applications.resnet50 import ResNet50
# model = ResNet50(weights='imagenet')
# model.save('')
print('Model loaded. Check http://127.0.0.1:5000/')

#
def model_predict(img_path, model):
    # def model_predict(img_path, model):
    data_dir_test = pathlib.Path("upload")

    # Define batch size, image height, and width
    batch_size = 1
    img_height = 228
    img_width = 228

    class_names = ['actinic keratosis', 'basal cell carcinoma', 'dermatofibroma', 'melanoma', 'nevus',
                   'pigmented benign keratosis', 'seborrheic keratosis', 'squamous cell carcinoma', 'vascular lesion']

    # Create the test dataset
    test_ds = tf.keras.preprocessing.image_dataset_from_directory(
        data_dir_test,
        # seed=123,
        image_size=(img_height, img_width),
        batch_size=batch_size)
    for images, true_labels in test_ds.take(1):  # Take the first batch of images
        predicted_class_indices = model.predict(images)  # Predict the labels for the images
        print("print 1")
        print(images.shape)
        print("print 2")
        print(true_labels)
        print("print 3")
        print(predicted_class_indices)

        true_label = class_names[true_labels[0].numpy()]
        print(true_label)

        predicted_label = class_names[np.argmax(predicted_class_indices[0])]
        print(predicted_label)
    return predicted_label


@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        # Get the file from post request
        f = request.files['file']

        # Save the file to ./uploads
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, 'upload', 'class', secure_filename(f.filename))
        # second_path = os.path.join(file_path, 'class', secure_filename(f.filename))
        f.save(file_path)

        # Make prediction

        result = model_predict(file_path, model)
        print(result)
        return result
    return None
#

if __name__ == '__main__':
    app.run(debug=True)
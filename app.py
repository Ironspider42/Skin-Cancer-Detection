from __future__ import division, print_function
# coding=utf-8
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
    # img = image.load_img(img_path, target_size=(228, 228))

    # # Preprocessing the image
    # x = image.img_to_array(img)
    # # x = np.true_divide(x, 255)
    # x = np.expand_dims(x, axis=0)
    #
    # # Be careful how your trained model deals with the input
    # # otherwise, it won't make correct prediction!
    # x = preprocess_input(x, mode='caffe')
    #
    # preds = model.predict(x)
    # return preds
    # img = image.load_img(img_path, target_size=(228, 228))

    img = load_img(img_path, target_size=(228, 228))
    img = img_to_array(img)
    img = img.reshape(1, 228, 228, 3)

    img = img.astype('float32')
    img = img / 255.0
    # img = tf.keras.preprocessing.image.load_img(img_path, target_size=(228, 228))
    # # img = Image.open(img_path)
    # image_array = tf.keras.preprocessing.image.img_to_array(img)
    # image_array = np.expand_dims(image_array, axis=0)
    # image_array = image_array / 255.0
    # # if img.mode != "RGB":
    #     img = img.convert("RGB")

    # Preprocessing the image
    # x = image.img_to_array(img)
    # print(image_array)
    # x = img.numpy().astype("uint8")
    # x = np.expand_dims(img, axis=0)
    # x = x / 255.0  # Rescale pixel values to [0, 1]

    # Make prediction
    # preds = model.predict(x)
    preds = model.predict(img)
    return preds


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
            basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)

        # Make prediction
        preds = model_predict(file_path, model)

        # Process your result for human
        # pred_class = preds.argmax(axis=-1)            # Simple argmax
        # pred_class = decode_predictions(preds, top=1)   # ImageNet Decode
        class_names = ['actinic keratosis', 'basal cell carcinoma', 'dermatofibroma', 'melanoma', 'nevus', 'pigmented benign keratosis', 'seborrheic keratosis', 'squamous cell carcinoma', 'vascular lesion']
        # predicted_label = class_names[np.argmax(predicted_class_indices[i])]
        print(preds)
        # idx = np.argmax(preds[0], axis=-1)
        idx = np.argmax(preds)
        print(idx)
        predicted_class_name = class_names[idx]
        # result = str(pred_class[0][0][1])               # Convert to string
        result = predicted_class_name
        print(class_names[np.argmax(preds)])
        return result
    return None
#

if __name__ == '__main__':
    app.run(debug=True)
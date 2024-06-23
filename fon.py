from __future__ import division, print_function
# coding=utf-8
import sys
import os
import glob
import re
import numpy as np
import pathlib
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
import matplotlib.pyplot as plt
from PIL import Image

print("TensorFlow version:", tf.__version__)

import keras
print("Keras version:", keras.__version__)
import PIL
# from keras.applications.imagenet_utils import decode_predictions
# Define a flask app
# app = Flask(__name__)

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
# #from keras.applications.resnet50 import ResNet50
# #model = ResNet50(weights='imagenet')
# #model.save('')
print('Model loaded. Check http://127.0.0.1:5000/')


# def model_predict(img_path, model):
data_dir_test = pathlib.Path("Test_images")

# Define batch size, image height, and width
batch_size = 32
img_height = 228
img_width = 228

class_names = ['actinic keratosis', 'basal cell carcinoma', 'dermatofibroma', 'melanoma', 'nevus', 'pigmented benign keratosis', 'seborrheic keratosis', 'squamous cell carcinoma', 'vascular lesion']

# Create the test dataset
test_ds = tf.keras.preprocessing.image_dataset_from_directory(
    data_dir_test,
    # seed=123,
    image_size=(img_height, img_width),
    batch_size=batch_size,

)
for images, true_labels in test_ds.take(1):  # Take the first batch of images
    predicted_class_indices = model.predict(images)  # Predict the labels for the images
    print("print 1")
    print(images.shape)
    print("print 2")
    print(true_labels)
    print("print 3")
    print(predicted_class_indices)

    plt.figure(figsize=(15, 10))
    for i in range(24):  # Display only 9 images to fit the subplot layout
        ax = plt.subplot(6, 4, i + 1)
        plt.imshow(images[i].numpy().astype("uint8"))
        true_label = class_names[true_labels[i].numpy()]
        print(true_label)
        predicted_label = class_names[np.argmax(predicted_class_indices[i])]
        print(predicted_label)
        plt.title(f"True: {true_label}\nPredicted: {predicted_label}")
        plt.axis("off")
plt.show()

# preds = model_predict('Test_images', model)


# print(preds)
# idx = np.argmax(preds)
# print(idx)
# predicted_class_name = class_names[idx]
# result = predicted_class_name
# print(class_names[np.argmax(preds)])


import os 
os.environ["KERAS_BACKEND"] = "torch"
import keras 

import numpy as np 
from helper import * 
from definitions import * 
import argparse


parser = argparse.ArgumentParser("Model") 
parser.add_argument("bundleName")
bundleName = parser.parse_args().bundleName

settings = load_settings(bundleName)
width, height = settings['main']['dimension'][0], settings['main']['dimension'][1]

training_input, training_label = load_dataset(bundleName, settings)

model = keras.Sequential(
    [
        keras.layers.InputLayer((height, width, 1), batch_size = None), 
        keras.layers.Convolution2D(3, (3,3), activation = keras.activations.leaky_relu, strides = (1,1)),
        keras.layers.Flatten(),
        keras.layers.Dense(3, keras.activations.leaky_relu),
        keras.layers.Dense(3, keras.activations.leaky_relu),
        keras.layers.Softmax()
    ]
)

model.compile("adam", keras.losses.CategoricalCrossentropy(False), metrics=["accuracy"]) 
model.fit(training_input, training_label, epochs=50)

model.save(f"./bundles/{bundleName}/model.keras")
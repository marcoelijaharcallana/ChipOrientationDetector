import os 
os.environ["KERAS_BACKEND"] = "torch"
import keras 

import numpy as np 
from helper import * 
from definitions import * 

training_input, training_label = load_dataset()

model = keras.Sequential(
    [
        keras.layers.InputLayer((280, 230, 1), batch_size = None), 
        keras.layers.Convolution2D(3, (3,3), activation = keras.activations.leaky_relu),
        keras.layers.MaxPool2D((2,2)),
        keras.layers.Convolution2D(3, (5,5), activation = keras.activations.leaky_relu),
        keras.layers.MaxPool2D((3,3)), 
        keras.layers.Convolution2D(3, (7,7), activation = keras.activations.leaky_relu),
        keras.layers.MaxPool2D((4,4)),
        keras.layers.Flatten(),
        keras.layers.Dense(3, keras.activations.leaky_relu),
        keras.layers.Dense(3, keras.activations.leaky_relu),
        keras.layers.Softmax()
    ]
)

model.compile("adam", keras.losses.CategoricalCrossentropy(False), metrics=["accuracy"]) 
model.fit(training_input, training_label, epochs=100)

model.save("model.keras")
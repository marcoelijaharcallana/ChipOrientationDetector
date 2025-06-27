import os 
os.environ["KERAS_BACKEND"] = "torch"
import keras 

import numpy as np 
from helper import * 
from definitions import * 

training_input, training_label = load_dataset()

model = keras.Sequential(
    [
        keras.layers.InputLayer((140, 115, 1), batch_size = None), 
        keras.layers.Convolution2D(3, (3,3), activation = keras.activations.leaky_relu, strides = (1,1)),
        keras.layers.Flatten(),
        keras.layers.Dense(3, keras.activations.leaky_relu),
        keras.layers.Dense(3, keras.activations.leaky_relu),
        keras.layers.Softmax()
    ]
)

model.compile("adam", keras.losses.CategoricalCrossentropy(False), metrics=["accuracy"]) 
model.fit(training_input, training_label, epochs=50)

model.save("model.keras")
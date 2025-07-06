import cv2 as cv 

import os 
os.environ["KERAS_BACKEND"] = "torch"

import keras 

from helper import * 
from definitions import *

import argparse

parser = argparse.ArgumentParser("AI")
parser.add_argument("bundleName")
bundleName = parser.parse_args().bundleName 

settings = load_settings(bundleName)

model = keras.models.load_model(f"bundles/{bundleName}/model.keras")
if model is None:
    print("Model not Found!")
    exit(1)

camera = cv.VideoCapture(0)
while camera.isOpened():
    image = get_image_from_camera_gray(camera)

    activated = check_guide_image(image, settings)
    main_image = get_main_image(image, settings)

    image = split_channel(image)
    image = add_overlay(image, settings)

    if activated:
        main_image = main_image / 255.0
        output = model(np.array([main_image]))[0].detach().cpu().numpy()

        print(output)

        highest, index = 0,0
        for i in range(3):
            if output[i] > highest:
                highest = output[i]
                index = i
        
        if index == 0:
            image = cv.putText(image, "PASS", (100,100), cv.FONT_HERSHEY_COMPLEX, 2.0, (0, 255, 0), 3)
        elif index == 1:
            image = cv.putText(image, "FAIL", (100,100), cv.FONT_HERSHEY_COMPLEX, 2.0, (0, 0, 255), 3)
        elif index == 2:
            pass
            #image = cv.putText(image, "NONE", (100,100), cv.FONT_HERSHEY_COMPLEX, 1.0, (0, 255, 255), 1)
    cv.imshow("Live Feed", image)
    
    key = cv.waitKey(20)
    if key == Key.Q:
        break 

cv.destroyAllWindows()
camera.release()
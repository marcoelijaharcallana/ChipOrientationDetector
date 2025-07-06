import cv2 as cv 
import json 
from helper import * 
from definitions import * 

import argparse 

parser = argparse.ArgumentParser("calibrate")
parser.add_argument("bundleName")
bundleName = parser.parse_args().bundleName
setup_directory(bundleName)


settings = load_settings(bundleName)

camera = cv.VideoCapture(0)
height, width = get_camera_resolution(camera)

def quit():
    cv.destroyAllWindows()
    camera.release()
    exit(0)


# Calibrate Guide Image
while True: 

    image = get_image_from_camera_gray(camera)
    guide_image = get_guide_image(image, settings)    

    image = split_channel(image)
    image = cv.line(image, (0,settings['guide']['position'][1]), (width, settings['guide']['position'][1]), (0,255,0), 1)
    image = add_overlay(image, settings)


    cv.imshow("Guide Calibration", image)   
    cv.imshow("Guide Image", guide_image) 
    input = cv.waitKey(20)

    if input == Key.Q:
        quit() 
    elif input == Key.SPACE:
        break 
    elif input == Key.UP:
        settings['guide']['position'][1] -= 1
    elif input == Key.DOWN:
        settings['guide']['position'][1] += 1
    elif input == Key.RIGHT:
        settings['guide']['position'][0] += 1
    elif input == Key.LEFT:
        settings['guide']['position'][0] -= 1
    elif input == Key.W:
        settings['guide']['radius'] += 1
    elif input == Key.S:
        settings['guide']['radius'] -= 1

cv.destroyAllWindows()


# Calibrate Main Image 
while True: 

    image = get_image_from_camera_gray(camera)
    main_image = get_main_image(image, settings)
    image = split_channel(image)    


    image = add_overlay(image, settings)


    input = cv.waitKey(20)

    if input == Key.Q:
        quit() 
    elif input == Key.SPACE:
        break 
    elif input == Key.UP:
        settings['main']['position'][1] -= 1
    elif input == Key.DOWN:
        settings['main']['position'][1] += 1
    elif input == Key.RIGHT:
        settings['main']['position'][0] += 1
    elif input == Key.LEFT:
        settings['main']['position'][0] -= 1
    elif input == Key.W: 
        settings['main']['dimension'][0] += 1
    elif input == Key.S: 
        settings['main']['dimension'][0] -=1
    elif input == Key.E: 
        settings['main']['dimension'][1] += 1
    elif input == Key.D:
        settings['main']['dimension'][1] -= 1

    cv.imshow("Main Window Calibration", image)
    cv.imshow("Main Image", main_image)

cv.destroyAllWindows()


# Calibrate Threshold  
while True: 

    image = get_image_from_camera_gray(camera)

    threshold = settings["threshold"]

    main_image = get_main_image(image, settings)
    brightness = get_guide_image_convolved_brightness(image, settings)
    decision = int(check_guide_image(image, settings))

    image = split_channel(image)    

    cv.putText(image, f"Brightness: {brightness}", (100,50), cv.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 1)
    cv.putText(image, f"Threshold: {threshold}", (100,100), cv.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 1)
    cv.putText(image, f"Verdict: {("None", "Detected")[decision]}", (100,150), cv.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 1)

    image = add_overlay(image, settings)
    
    cv.imshow("Calibrate Threshold", image)


    input = cv.waitKey(1)

    if input == Key.Q:
        quit() 
    elif input == Key.SPACE:
        break 
    elif input == Key.UP:
        settings['threshold'] += 1
    elif input == Key.DOWN:
        settings['threshold'] -= 1

# Save JSON 
save_settings(bundleName, settings)

import cv2 as cv 
from definitions import * 
import json
import numpy as np 

import os 
import sys 

# Gray to BGR Wrapper
def split_channel(image):
    return cv.cvtColor(image, cv.COLOR_GRAY2BGR)

# Read Image from Camera with Safety Checks and Apply Median Blur
def get_image_from_camera(camera): 
    if camera.isOpened():
        ret, image = camera.read()
        image = cv.medianBlur(image, 3)
        if ret:
            return image 
        else:
            raise Exception(Error.CAMERA_ERROR_READ,"Error Reading Camera")
    else:
        raise Exception(Error.CAMERA_NOT_FOUND,"Camera not Found")

# Read Image from Camera with Safety Checks and make it Grayscale (HAS BGR CHANNEL)
def get_image_from_camera_gray_3ch(camera):
    return cv.cvtColor(cv.cvtColor(get_image_from_camera(camera), cv.COLOR_BGR2GRAY), cv.COLOR_GRAY2BGR)

 
# Read Image from Camera with Safety Checks and make it Grayscale (HAS GRAY CHANNEL ONLY)
def get_image_from_camera_gray(camera):
    return cv.cvtColor(get_image_from_camera(camera), cv.COLOR_BGR2GRAY)


def load_dataset(path, settings):
    if os.path.exists(path):
        data = np.load(path)

        return data['input'], data['label']
    else:
        return np.empty((0,settings['main']['dimension'][1],settings['main']['dimension'][0])), np.empty((0,3)) # TODO adjust to image size


def save_dataset(input, label, path):
    np.savez(path, input=input, label=label)

def crop_image_pp(image, p1, p2):
    x1, y1 = p1 
    x2, y2 = p2 

    x1, x2 = min(x1,x2), max(x1,x2)
    y1, y2 = min(y1,y2), max(y1,y2)

    return image[y1:y2+1, x1:x2+1]

def crop_image_pd(image, p, width, height):
    x,y = p 
    return image[y:y+height,x:x+width]

 
def get_guide_image(image, settings):
    radius = settings['guide']['radius']
    position = settings['guide']['position']

    if get_guide_image.pradius != radius:
        get_guide_image.guide_mask = np.zeros((2 * radius + 1, 2 * radius + 1), "uint8")
        get_guide_image.guide_mask = cv.circle(get_guide_image.guide_mask , (radius,radius), radius, (255, 255, 255), -1)
        get_guide_image.pradius = radius 
    
    image = crop_image_pp(image, (position[0] - radius,  position[1] - radius), (position[0] + radius, position[1] + radius))
    image = cv.bitwise_and(image, get_guide_image.guide_mask)
    return image 

get_guide_image.guide_mask = None
get_guide_image.pradius = 0


def get_guide_image_brightness(image, settings):
    radius = settings['guide']['radius']
    image = get_guide_image(image, settings)

    dv,sm = 0,0
    for i in range(0, 2 * radius + 1):
        for j in range(0, 2 * radius + 1):
            dv += get_guide_image.guide_mask[i,j] == 255
            sm += int(image[i,j])
            
    return sm / dv

def get_guide_image_convolved_brightness(image, settings):
    radius = settings['guide']['radius']
    
    image = get_guide_image(image, settings)
    image = cv.blur(image, (radius, radius)) 
    _, mx, _, _ = cv.minMaxLoc(image)

    return mx

def check_guide_image(image, settings):
    brightness = get_guide_image_convolved_brightness(image, settings)
    return brightness > settings["threshold"]

def get_main_image(image, settings):
    return crop_image_pd(image, settings['main']['position'], settings['main']['dimension'][0], settings['main']['dimension'][1])


def get_camera_resolution(camera):
    img = get_image_from_camera(camera)
    return img[:,:,0].shape

# Load Settings JSON and Perform Sanity Checks 
def load_settings():
    fp = open("settings.json")
    settings = json.load(fp)
    fp.close()
    
    guide = settings['guide']
    radius = guide['radius']
    position = guide['position']

    main = settings['main']
    position = main['position']
    dimension = main['dimension']
    
    return settings 


def save_settings(settings):
    fp = open('settings.json', 'w')
    json.dump(settings, fp)
    fp.close()

def add_overlay(img, settings):
    guide_position = settings['guide']['position']
    guide_radius = settings['guide']['radius']
    guide_color = settings['guide']['color']

    img = cv.circle(img, guide_position, guide_radius, guide_color, 1)

    main_position = settings['main']['position']
    main_dimension = settings['main']['dimension']
    main_color = settings['main']['color']
    main_width = main_dimension[0]
    main_height = main_dimension[1]

    img = cv.rectangle(img, (main_position[0], main_position[1], main_width, main_height), main_color, 1)  

    return img 
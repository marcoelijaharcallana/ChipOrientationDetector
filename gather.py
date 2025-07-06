import cv2 as cv 
from definitions import *
from helper import *
import argparse 

parser = argparse.ArgumentParser("Gather")
parser.add_argument("trainingFile")
arguments = parser.parse_args()


camera = cv.VideoCapture(0) 
settings = load_settings()
training_input, training_label = load_dataset(arguments.trainingFile, settings)
backup_training_input, backup_training_label = training_input, training_label 

mode = "NOT SET"

ps, nn, fl = 0,0,0

while camera.isOpened():
    image = get_image_from_camera_gray(camera)

    main_image = get_main_image(image, settings)
    
    key = cv.waitKey(20)
    if key == Key.Q:
        break 
    elif key == Key.Z:
        if check_guide_image(image, settings):
            training_input = np.concatenate((training_input, [main_image / 255.0]))
            training_label = np.concatenate((training_label ,[[1,0,0]]))  
            ps += 1 
    elif key == Key.X:
        if check_guide_image(image, settings):
            training_input = np.concatenate((training_input, [main_image / 255.0]))
            training_label = np.concatenate((training_label ,[[0,1,0]]))  
        fl += 1
    elif key == Key.C:        
        if check_guide_image(image, settings):
            training_input = np.concatenate((training_input, [main_image / 255.0]))
            training_label = np.concatenate((training_label ,[[0,0,1]]))  
            nn += 1
    elif key == Key.S: 
        save_dataset(arguments.trainingFile, training_input, training_label)

        print("ADDED ", training_label.shape[0] - backup_training_label.shape[0]) 
        backup_trainng_input, backup_training_label = training_input, training_label

        print("SAVED")
    elif key == Key.R:
        training_input, training_label = backup_training_input, backup_training_label 

        print("REVERTED")
    else:
        mode = "NOT SET"

    image = split_channel(image)
    image = add_overlay(image, settings)

    image = cv.putText(image, f"Pass: {ps} Fail: {fl} None: {nn}", (100,100), cv.FONT_HERSHEY_COMPLEX, 0.5, (0,255,0), 1)

    cv.imshow("Gather Data", image)
    cv.imshow("Main Image", main_image)

camera.release()
cv.destroyAllWindows() 
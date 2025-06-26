import cv2 as cv 
from definitions import *
from helper import *

training_input, training_label = load_dataset()
backup_training_input, backup_training_label = training_input, training_label 
camera = cv.VideoCapture(0) 
settings = load_settings()
mode = "NOT SET"

while camera.isOpened():
    image = get_image_from_camera_gray(camera)

    main_image = get_main_image(image, settings)
    
    key = cv.waitKey(20)
    if key == Key.Q:
        break 
    elif key == Key.Z:
        mode = "PASS"
    elif key == Key.X:
        mode = "FAIL"
    elif key == Key.C:        
        mode = "NONE"
    elif key == Key.S: 
        save_dataset(training_input, training_label)

        print("ADDED ", training_label.shape[0] - backup_training_label.shape[0]) 
        backup_trainng_input, backup_training_label = training_input, training_label

        print("SAVED")
    elif key == Key.R:
        training_input, training_label = backup_training_input, backup_training_label 

        print("REVERTED")
    else:
        mode = "NOT SET"


    if check_guide_image(image, settings):
        if mode == "PASS":
            training_input = np.concatenate((training_input, [main_image / 255.0]))
            training_label = np.concatenate((training_label ,[[1,0,0]]))            
        elif mode == "FAIL":
            training_input = np.concatenate((training_input, [main_image / 255.0]))
            training_label = np.concatenate((training_label ,[[0,1,0]]))            
        elif mode == "NONE":
            training_input = np.concatenate((training_input, [main_image / 255.0]))
            training_label = np.concatenate((training_label ,[[0,0,1]]))            

    image = split_channel(image)
    image = add_overlay(image, settings)
    image = cv.putText(image, f"Mode: {mode}", (100,100), cv.FONT_HERSHEY_COMPLEX, 1, (0,255,0), 1)

    cv.imshow("Gather Data", image)
    cv.imshow("Main Image", main_image)

camera.release()
cv.destroyAllWindows() 
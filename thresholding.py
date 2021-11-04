import numpy as np
import cv2
import os

def perform_thresholding(args, image_name, image):
    '''
    Args:
        image: Non maxima suppressed
    Returns:
        img1 : Image after applying threshold t1
        img2 : Image after applying threshold t2
        img3 : Image after applying threshold t3
    '''

    # Store all the values of image after non- maxima suppression which are greater than zero into array
    image_arr = image[image>0].ravel()

    # Get 25th percentile of the array
    t1 = np.percentile(image_arr,25)
    cv2.imwrite(os.path.join(args.output_folder, image_name + '_threshold_t1.bmp'), (image > t1).astype("int32"))

    # Get 50th percentile of the array
    t2 = np.percentile(image_arr,50)
    cv2.imwrite(os.path.join(args.output_folder, image_name + '_threshold_t2.bmp'), (image > t2).astype("int32"))

    # Get 75th percentile of the array
    t3 = np.percentile(image_arr,75)
    cv2.imwrite(os.path.join(args.output_folder, image_name + '_threshold_t3.bmp'), (image > t2).astype("int32"))
    
    # Apply threshold to the image and convert it into integer array
    return (image > t1).astype("int32"), (image > t2).astype("int32"), (image > t3).astype("int32") 


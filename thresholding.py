import os

import numpy as np
import cv2

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
    image_1 = (image > t1).astype("int32")
    cv2.imwrite(os.path.join(args.output_folder, image_name + f'_threshold_t1_{np.round(t1, 2)}.bmp'), image_1 * 255) # Multiplying the image with 255 for contrast

    # Get 50th percentile of the array
    t2 = np.percentile(image_arr,50)
    image_2 = (image > t2).astype("int32")
    cv2.imwrite(os.path.join(args.output_folder, image_name + f'_threshold_t2_{np.round(t2, 2)}.bmp'), image_2 * 255)

    # Get 75th percentile of the array
    t3 = np.percentile(image_arr,75)
    image_3 = (image > t3).astype("int32")
    cv2.imwrite(os.path.join(args.output_folder, image_name + f'_threshold_t3_{np.round(t3, 2)}.bmp'), image_3 * 255)
    
    # Apply threshold to the image and convert it into integer array
    return image_1, image_2, image_3 


'''
Computer Vision Midterm Project
Project group members:  
    1. Aniket Bote (N12824308)
    2. Sindhu Harish (N19806874)
'''

import os

import cv2
import numpy as np

from utils import Operator, apply_discrete_convolution


def perform_gaussian_smoothing(args, image_name, image):
    '''
    Args:
        image : An image to on which smoothing will appear
    Returns:
        smoothened image : Smoothened image
    '''
    # Apply discreet convolution with gaussian mask
    image = apply_discrete_convolution(image, Operator.gaussian_mask)

    # Normalize the image
    image = image / np.sum(Operator.gaussian_mask)
 
    #write image into the output folder after normalization
    cv2.imwrite(os.path.join(args.output_folder, image_name + '_gaussian_smooth_normalized.bmp'), image)

    # Return the smoothened image
    return image

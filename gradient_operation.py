import os

import cv2
import numpy as np

from utils import Operator, apply_discrete_convolution


def perform_gradient_operation(args, image_name, image):
    '''
    Args:
        image : An image on which gradient operation will happen
    Returns:
        Magnitude : Magnitude of the gradient
        Theta     : Gradient Angle
    '''
    # Compute horizontal gradients
    dfdx = apply_discrete_convolution(image, Operator.gx)

    #Copy Image to Output folder after horizontal gradient
    cv2.imwrite(os.path.join(args.output_folder, image_name + '_Gx_normalized.bmp'), dfdx)

    # Compute vertical gradients
    dfdy = apply_discrete_convolution(image, Operator.gy)

    #Copy Image to Output folder after vertical gradient
    cv2.imwrite(os.path.join(args.output_folder, image_name + '_Gy_normalized.bmp'), dfdy)

    # Compute magnitude of the gradient
    m = np.sqrt(np.square(dfdx) + np.square(dfdy))
    
    # Normalize gradient magnitude
    m = np.absolute(m) / 3

    #Copy Image to Output folder with gradient magnitude value
    cv2.imwrite(os.path.join(args.output_folder, image_name + '_gradient_magnitude_normalized.bmp'), m)

    # Compute gradient angle
    theta = np.degrees(np.arctan2(dfdy, dfdx))

    return m, theta

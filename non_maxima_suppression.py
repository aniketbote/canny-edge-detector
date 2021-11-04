import numpy as np
from utils import get_positive_angle, Sector
import os

def perform_non_maxima_suppression(args, image_name, magnitude, gradient_angle):
    '''
    Args:
        magnitude : Magnitude of the gradient
        gradient_angle : Gradient angle
    Returns:
        Magnitude : Magnitude array after non-maxima supression
    '''
    # Compute positive angles
    positive_gradient_angle = get_positive_angle(gradient_angle)

    # Get magnitude array shape
    m_arr, n_arr = magnitude.shape

    # reference pixel location during start of the process
    rpi_m, rpi_n = 1,1

    # Build output array
    output_arr = np.ones((m_arr , n_arr)) * np.nan

    for i in range(m_arr - 2):
        for j in range(n_arr - 2):
            # Compute output pixel location for output array
            op_m, op_n = i + rpi_m, j + rpi_n

            # Get 3 x 3 magnitude slice
            arr_slice = magnitude[i:i+3, j:j+3]

            # Get 3 x 3 angle slice
            angle_slice = positive_gradient_angle[i:i+3, j:j+3]

            # If undefined value at reference pixel in magnitude or angle put zero in output pixel location
            if np.isnan(arr_slice[rpi_m][rpi_n]) or np.isnan(angle_slice[rpi_m][rpi_n]):
                output_arr[op_m][op_n] = 0
            else:
                # Get the sector value
                sector = Sector().get_sector(angle_slice[rpi_m][rpi_n])
                
                if sector == 0:
                    # If undefined value at any of sector neighbour put zero in output pixel location
                    if np.isnan(arr_slice[rpi_m][rpi_n+1]) or np.isnan(arr_slice[rpi_m][rpi_n-1]):
                        output_arr[op_m][op_n] = 0

                    # If reference pixel is greater than its sector neighbours put reference pixel value at output location
                    elif arr_slice[rpi_m][rpi_n] > arr_slice[rpi_m][rpi_n+1] and arr_slice[rpi_m][rpi_n] > arr_slice[rpi_m][rpi_n-1]:
                        output_arr[op_m][op_n] = arr_slice[rpi_m][rpi_n]

                    # If reference pixel value is less than its sector neighbours put zero in output pixel location
                    else:
                        output_arr[op_m][op_n] = 0
 
                elif sector == 1:
                    # If undefined value at any of sector neighbour put zero in output pixel location
                    if np.isnan(arr_slice[rpi_m-1][rpi_n+1]) or np.isnan(arr_slice[rpi_m+1][rpi_n-1]):
                        output_arr[op_m][op_n] = 0

                    # If reference pixel is greater than its sector neighbours put reference pixel value at output location
                    elif arr_slice[rpi_m][rpi_n] > arr_slice[rpi_m-1][rpi_n+1] and arr_slice[rpi_m][rpi_n] > arr_slice[rpi_m+1][rpi_n-1]:
                        output_arr[op_m][op_n] = arr_slice[rpi_m][rpi_n]

                    # If reference pixel value is less than its sector neighbours put zero in output pixel location
                    else:
                        output_arr[op_m][op_n] = 0

                elif sector == 2:
                    # If undefined value at any of sector neighbour put zero in output pixel location
                    if np.isnan(arr_slice[rpi_m-1][rpi_n]) or np.isnan(arr_slice[rpi_m+1][rpi_n]):
                        output_arr[op_m][op_n] = 0

                    # If reference pixel is greater than its sector neighbours put reference pixel value at output location
                    elif arr_slice[rpi_m][rpi_n] > arr_slice[rpi_m-1][rpi_n] and arr_slice[rpi_m][rpi_n] > arr_slice[rpi_m+1][rpi_n]:
                        output_arr[op_m][op_n] = arr_slice[rpi_m][rpi_n]

                    # If reference pixel value is less than its sector neighbours put zero in output pixel location
                    else:
                        output_arr[op_m][op_n] = 0

                elif sector == 3:
                    # If undefined value at any of sector neighbour put zero in output pixel location
                    if np.isnan(arr_slice[rpi_m-1][rpi_n-1]) or np.isnan(arr_slice[rpi_m+1][rpi_n+1]):
                        output_arr[op_m][op_n] = 0

                    # If reference pixel is greater than its sector neighbours put reference pixel value at output location
                    elif arr_slice[rpi_m][rpi_n] > arr_slice[rpi_m-1][rpi_n-1] and arr_slice[rpi_m][rpi_n] > arr_slice[rpi_m+1][rpi_n+1]:
                        output_arr[op_m][op_n] = arr_slice[rpi_m][rpi_n]

                    # If reference pixel value is less than its sector neighbours put zero in output pixel location
                    else:
                        output_arr[op_m][op_n] = 0

                # If sector value is other 0,1,2,3 raise an error.(Not going to happen its there for correctness)
                else:
                    raise f"Undefined sector: {sector}"           
    return output_arr

# Import the required libraries
import numpy as np


# A class to store all operators
class Operator:
    # Prewitt operator for Gx
    gx = np.array([
        [-1,0,1],
        [-1,0,1],
        [-1,0,1]])
    
    # Prewitt operator for Gy
    gy = np.array([
        [1,1,1],
        [0,0,0],
        [-1,-1,-1]])

    # Gaussian mask
    gaussian_mask = np.array([
        [1,1,2,2,2,1,1],
        [1,2,2,4,2,2,1],
        [2,2,4,8,4,2,2],
        [2,4,8,16,8,4,2],
        [2,2,4,8,4,2,2],
        [1,2,2,4,2,2,1],
        [1,1,2,2,2,1,1]])

# A class to store sector angle definitions and method to provide sector based on angle
class Sector():
    def __init__(self):
        # Dictionary with {sector: sector range}
        self.sector = {0: [(0, 22.5),(337.5,360),(157.5,202.5)], 1: [(22.5,67.5), (202.5,247.5)], 2:[(67.5,112.5), (247.5, 292.5)], 3:[(112.5, 157.5), (292.5,337.5)]}

    def get_sector(self, angle):
        for key, val in self.sector.items():
            for l,u in val:
                # check if angle lies in the range if yes return key
                if angle >= l and angle < u:
                    return key
        # If angle is not in any range we return -1. (Not going to happen. Its there for correctness)
        return -1

# A function to apply dicreet convolutions
def apply_discrete_convolution(image, mask):
    '''
    Args:
        image : An image to use for convolution
        mask  : An mask to use for convolution
    Returns:
        convolved image: An image after convolution
    '''
    # Get the shape of image and mask
    (m_image, n_image), (m_mask, n_mask) = image.shape, mask.shape

    # Compute the reference pixel index from where output array will start populating
    rpi_m, rpi_n = int(np.floor(m_mask/2)), int(np.floor(n_mask/2))

    # Initialize an output array with nan values
    output_arr = np.ones((m_image, n_image)) * np.nan

    # Iterate through the image
    for i in range(m_image - m_mask + 1):
        for j in range(n_image - n_mask + 1):
            # Isolate the image slice to apply convolution
            img_slice = image[i:i+m_mask, j:j+n_mask]
            # Apply convolution and store the result in output array in approriate location
            output_arr[i+rpi_m][j+rpi_n] = np.sum(img_slice * mask)

    return output_arr

# A function to convert negative angles to positive angles
def get_positive_angle(angle):
    pos_angle = angle.copy()
    pos_angle[pos_angle<0] += 360
    return pos_angle


# A funtion to perform gaussian smoothing
def perform_gaussian_smoothing(image):
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

    # Return the smoothened image
    return image


def perform_gradient_operation(image):
    '''
    Args:
        image : An image on which gradient operation will happen
    Returns:
        Magnitude : Magnitude of the gradient
        Theta     : Gradient Angle
    '''
    # Compute horizontal gradients
    dfdx = apply_discrete_convolution(image, Operator.gx)

    # Compute vertical gradients
    dfdy = apply_discrete_convolution(image, Operator.gy)

    # Compute magnitude of the gradient
    m = np.sqrt(np.square(dfdx) + np.square(dfdy))

    # Normalize gradient magnitude
    m = np.absolute(m) / 3

    # Compute gradient angle
    theta = np.degrees(np.arctan2(dfdy, dfdx))

    return m, theta


def perform_non_maxima_suppression(magnitude, gradient_angle):
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


def perform_thresholding(image):
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

    # Get 50th percentile of the array
    t2 = np.percentile(image_arr,50)

    # Get 75th percentile of the array
    t3 = np.percentile(image_arr,75)
    
    # Apply threshold to the image and convert it into integer array
    return (image > t1).astype("int32"), (image > t2).astype("int32"), (image > t3).astype("int32") 













if __name__ == "__main__":
    import argparse
    import matplotlib.pyplot as plt
    from PIL import Image

    parser=argparse.ArgumentParser()
    parser.add_argument('--input_image', '-i', help='Image path')
    args = parser.parse_args()
    # image_path = ''
    if args.input_image == None:
        image_path = input("Enter image path:\n")
    else:
        image_path = args.input_image
    # Reading the image
    print("Reading the image")
    image = np.array(Image.open(image_path))

    print("Performing gaussian smoothing")
    gaussian_smooth_image = perform_gaussian_smoothing()
    




    # print("Running")
    # test_arr = np.array([[1, 1, 1, 1, 1, 1, 5],
    #             [1, 1, 1, 1, 1, 5, 9],
    #             [1, 1, 1, 1, 5, 9, 9],
    #             [1, 1, 1, 5, 9, 9, 9],
    #             [1, 1, 5, 9, 9, 9, 9],
    #             [1, 5, 9, 9, 9, 9, 9],
    #             [5, 9, 9, 9, 9, 9, 9]])
    # M, THETA = perform_gradient_operation(test_arr)
    # print(M)
    # print(THETA)
    # NMS = perform_non_maxima_suppression(M, THETA)
    # print(NMS)
    # T1, T2, T3 = perform_thresholding(NMS)
    # print(T1)

    

    # g_img = perform_gaussian_smoothing(test_image)
    # M, THETA = perform_gradient_operation(g_img)
    # NMS = perform_non_maxima_suppression(M, THETA)
    # T1, T2, T3 = perform_thresholding(NMS)
    # plt.imshow(T1, cmap = "gray")
    # plt.show()
    # plt.imshow(T2, cmap = "gray")
    # plt.show()
    # plt.imshow(T3, cmap = "gray")
    # plt.show()
    
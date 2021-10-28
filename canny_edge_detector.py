# Import the required libraries
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

# A class to store all operators
class Operator:
    gx = np.array([
        [-1,0,1],
        [-1,0,1],
        [-1,0,1]])
    gy = np.array([
        [1,1,1],
        [0,0,0],
        [-1,-1,-1]])
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
        self.sector = {0: [(0, 22.5),(337.5,360),(157.5,202.5)], 1: [(22.5,67.5), (202.5,247.5)], 2:[(67.5,112.5), (247.5, 292.5)], 3:[(112.5, 157.5), (292.5,337.5)]}

    def get_sector(self, angle):
        for key, val in self.sector.items():
            for l,u in val:
                if angle >= l and angle < u:
                    return key
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

    # Compute gradient angle
    theta = np.degrees(np.arctan2(dfdy, dfdx))

    return m, theta

def perform_non_maxima_suppression(magnitude, gradient_angle):
    positive_gradient_angle = get_positive_angle(gradient_angle)
    m_arr, n_arr = magnitude.shape
    rpi_m, rpi_n = 1,1
    output_arr = np.ones((m_arr , n_arr)) * np.nan
    for i in range(m_arr - 2):
        for j in range(n_arr - 2):
            op_m, op_n = i + rpi_m, j + rpi_n
            arr_slice = magnitude[i:i+3, j:j+3]
            angle_slice = positive_gradient_angle[i:i+3, j:j+3]
            if np.isnan(arr_slice[rpi_m][rpi_n]) or np.isnan(angle_slice[rpi_m][rpi_n]):
                output_arr[op_m][op_n] = 0
            else:
                sector = Sector().get_sector(angle_slice[rpi_m][rpi_n])
                if sector == 0:
                    if np.isnan(arr_slice[rpi_m][rpi_n+1]) or np.isnan(arr_slice[rpi_m][rpi_n-1]):
                        output_arr[op_m][op_n] = 0
                    elif arr_slice[rpi_m][rpi_n] > arr_slice[rpi_m][rpi_n+1] and arr_slice[rpi_m][rpi_n] > arr_slice[rpi_m][rpi_n-1]:
                        output_arr[op_m][op_n] = arr_slice[rpi_m][rpi_n]
                    else:
                        output_arr[op_m][op_n] = 0
                elif sector == 1:
                    if np.isnan(arr_slice[rpi_m-1][rpi_n+1]) or np.isnan(arr_slice[rpi_m+1][rpi_n-1]):
                        output_arr[op_m][op_n] = 0
                    elif arr_slice[rpi_m][rpi_n] > arr_slice[rpi_m-1][rpi_n+1] and arr_slice[rpi_m][rpi_n] > arr_slice[rpi_m+1][rpi_n-1]:
                        output_arr[op_m][op_n] = arr_slice[rpi_m][rpi_n]
                    else:
                        output_arr[op_m][op_n] = 0
                elif sector == 2:
                    if np.isnan(arr_slice[rpi_m-1][rpi_n]) or np.isnan(arr_slice[rpi_m+1][rpi_n]):
                        output_arr[op_m][op_n] = 0
                    elif arr_slice[rpi_m][rpi_n] > arr_slice[rpi_m-1][rpi_n] and arr_slice[rpi_m][rpi_n] > arr_slice[rpi_m+1][rpi_n]:
                        output_arr[op_m][op_n] = arr_slice[rpi_m][rpi_n]
                    else:
                        output_arr[op_m][op_n] = 0
                elif sector == 3:
                    if np.isnan(arr_slice[rpi_m-1][rpi_n-1]) or np.isnan(arr_slice[rpi_m+1][rpi_n+1]):
                        output_arr[op_m][op_n] = 0
                    elif arr_slice[rpi_m][rpi_n] > arr_slice[rpi_m-1][rpi_n-1] and arr_slice[rpi_m][rpi_n] > arr_slice[rpi_m+1][rpi_n+1]:
                        output_arr[op_m][op_n] = arr_slice[rpi_m][rpi_n]
                    else:
                        output_arr[op_m][op_n] = 0
                else:
                    raise f"Undefined sector: {sector}"           
    return output_arr


def perform_thresholding(image):
    image_arr = image[image>0].ravel()
    t1 = np.percentile(image_arr,25)
    t2 = np.percentile(image_arr,50)
    t3 = np.percentile(image_arr,75)
    
    return (image > t1).astype("int32"), (image > t2).astype("int32"), (image > t3).astype("int32") 













if __name__ == "__main__":
    print("Running")
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
    test_image = np.array(Image.open('lena512.bmp'))
    g_img = perform_gaussian_smoothing(test_image)
    M, THETA = perform_gradient_operation(g_img)
    NMS = perform_non_maxima_suppression(M, THETA)
    T1, T2, T3 = perform_thresholding(NMS)
    plt.imshow(T1, cmap = "gray")
    plt.show()
    plt.imshow(T2, cmap = "gray")
    plt.show()
    plt.imshow(T3, cmap = "gray")
    plt.show()
    
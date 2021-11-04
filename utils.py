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

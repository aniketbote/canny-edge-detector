import numpy as np

from gaussian_smoothing import perform_gaussian_smoothing
from gradient_operation import perform_gradient_operation
from non_maxima_suppression import perform_non_maxima_suppression
from thresholding import perform_thresholding

test_arr = np.array([[1, 1, 1, 1, 1, 1, 5],
            [1, 1, 1, 1, 1, 5, 9],
            [1, 1, 1, 1, 5, 9, 9],
            [1, 1, 1, 5, 9, 9, 9],
            [1, 1, 5, 9, 9, 9, 9],
            [1, 5, 9, 9, 9, 9, 9],
            [5, 9, 9, 9, 9, 9, 9]])

class TestArgument:
    input_folder = 'input'
    output_folder = 'output'

args = TestArgument

# g_img = perform_gaussian_smoothing(test_image)
M, THETA = perform_gradient_operation(args, 'test_arr',test_arr)
print(M)
print(THETA)
NMS = perform_non_maxima_suppression(args, 'test_arr', M, THETA)
print(NMS)
T1, T2, T3 = perform_thresholding(args, 'test_arr',NMS)
print(T1)
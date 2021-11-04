# Import the required libraries
import argparse
import glob
import os
import shutil

import cv2

from gaussian_smoothing import perform_gaussian_smoothing
from gradient_operation import perform_gradient_operation
from non_maxima_suppression import perform_non_maxima_suppression
from thresholding import perform_thresholding


parser = argparse.ArgumentParser()
parser.add_argument(
    '--input_folder',
    type=str,
    default='input',
    required=False,
    help='input folder with images'                        
)

parser.add_argument(
    '--output_folder',
    type=str,
    default='output',
    required=False,
    help='output folder to save processed images'                        
)

args = parser.parse_args()

if os.path.exists(args.output_folder):
    shutil.rmtree(args.output_folder)

os.makedirs(args.output_folder)

print("Reading images from input folder")
images = glob.glob(os.path.join(args.input_folder, '*.bmp'))
for image_name in images:
    output_image_name = image_name.split('\\')[1].split('.bmp')[0]
    img = cv2.imread(image_name, cv2.IMREAD_GRAYSCALE)   

    print("Performing gaussian smoothing for image: " + output_image_name)
    gaussian_smooth_image = perform_gaussian_smoothing(args, output_image_name, img)

    print("Performing gradient smoothing for image: " + output_image_name)
    M, THETA = perform_gradient_operation(args, output_image_name, gaussian_smooth_image)

    print("Performing non-maxima suppression for image: " + output_image_name)
    NMS = perform_non_maxima_suppression(args, output_image_name, M, THETA)

    print("Performing thresholding for image: " + output_image_name, '\n')
    T1, T2, T3 = perform_thresholding(args, output_image_name, NMS)



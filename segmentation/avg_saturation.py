# for visualizations
import numpy as np
from IPython.display import clear_output
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import cv2


### optional ## just images in white flash / daylight 
# Find the average saturation of an image
def avg_saturation(rgb_image):
    # Convert image to HSV
    hsv = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2HSV)

    # Add up all the pixel values in the V channel
    sum_brightness = np.sum(hsv[:,2,0])
    area = 600*1100.0  # pixels
    
    # find the avg
    avg = sum_brightness/area
    
    return avg
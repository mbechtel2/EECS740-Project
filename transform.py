# Given an image, filter it using an approach based on the Grey-Edge Hypothesis

import numpy as np
from skimage import filters, io, color, exposure
import time
import cv2

def grey_edge(img, minkowski, sigma):
    # Convert the image to the Grayscale space
    new_img = color.rgb2gray(img)
    
    # Apply a Gaussian filter to the image
    new_img = filters.gaussian(new_img, sigma, multichannel=True)
    
    # Compute and apply a Grey-Edge filter to the image
    greyedges = (np.sum((np.abs(filters.sobel(new_img)**minkowski))))**(1/minkowski)
    new_img /= greyedges
    
    # Convert the image to have 3 channels (required by CNN architecture)
    #   Image will still resemble a Grayscale image
    new_img = color.gray2rgb(new_img)
    
    # Return the Grey-Edge filtered image
    return new_img
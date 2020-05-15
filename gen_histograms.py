# Create histograms for the simulated test lighting scenarios#
#
# This program is meant to replicate Figures 1a-1f in the report

import numpy as np
from skimage import io,exposure,color
import matplotlib.pyplot as plt
import cv2

# Open and calculate the histogram for an image from the original dataset
vid = cv2.VideoCapture("epoch/test/Car_test.avi")
ret,img = vid.read() 
cv2.imwrite("figs/fig1a.png", img)
#img = color.rgb2gray(img)* 255
plt.hist(img.flatten(), 256, range=(0,256))
plt.xlabel("Pixel Intensity")
plt.ylabel("Number of Pixels")
plt.savefig("figs/fig1d.pdf")

# Open and calculate the histogram for an image from the dark simulated dataset
vid = cv2.VideoCapture("epoch/test/Car_test_gamma4.avi")
ret,img = vid.read() 
cv2.imwrite("figs/fig1b.png", img)
#img = color.rgb2gray(img)* 255
plt.hist(img.flatten(), 256, range=(0,256))
plt.xlabel("Pixel Intensity")
plt.ylabel("Number of Pixels")
plt.savefig("figs/fig1e.pdf")

# Open and calculate the histogram for an image from the light simulated dataset
vid = cv2.VideoCapture("epoch/test/Car_test_gamma0.avi")
ret,img = vid.read() 
cv2.imwrite("figs/fig1c.png", img)
#img = color.rgb2gray(img)* 255
plt.hist(img.flatten(), 256, range=(0,256))
plt.xlabel("Pixel Intensity")
plt.ylabel("Number of Pixels")
plt.savefig("figs/fig1f.pdf")
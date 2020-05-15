# Create Spectrum images to use for Histogram Matching
#
# Note the Spectrum images created are represented as Figures 3a-3d and their 
#   histograms are represented as Figures 3e-3h in the report
#
# Example usage:
#   python create_spectrums 320 240 4

import sys,os
import skimage
from skimage import exposure,io
from skimage.exposure import histogram
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

# Get image parameters
width = int(sys.argv[1]) # Image width
height = int(sys.argv[2]) # Image height
size = width * height # Total number of pixels
num_spectrums = int(sys.argv[3]) # Number of Spectrums to create
num_colors = int(256 / num_spectrums) # Number of colors/intensities in each Spectrum
color_pixels = int(size / num_colors) # Number of pixels per color for equal histogram

# Create folder for storing histograms
folder_name = "{}x{}_{}spec".format(width, height, num_spectrums)
if not os.path.isdir(folder_name):
    os.mkdir(folder_name)

# Variables used for creating Spectrums
color = 0
num_color = 0
xvals = np.arange(256)

# For each Spectrum
for s in range(num_spectrums): 
    # Create an image
    img = []
    for i in range(height):
        img.append([])
        for j in range(width):
            # Add a new pixel of the current color
            img[i].append([])
            img[i][j].append(color)
            img[i][j].append(color)
            img[i][j].append(color)
            num_color += 1
            
            # If the desired number of pixels for the current color is reached
            #   Move to the next highest color
            if num_color == color_pixels - 1:
                if not color == 255:
                    color += 1
                num_color = 0            
    
    # Save the image and it's corresponding histogram
    img = np.asarray(img, dtype=np.uint8)
    io.imsave("{}/spec{}.jpg".format(folder_name, s), img)
    plt.figure()
    plt.hist(img.flatten(), 256, range=(0,256))
    plt.xlabel("Pixel Intensity")
    plt.ylabel("Number of Pixels")
    plt.savefig("{}/spec{}_hist.pdf".format(folder_name,s))
    
    to_pdf = Image.open("{}/spec{}.jpg".format(folder_name, s))
    to_pdf.save("{}/spec{}.pdf".format(folder_name, s))
    to_pdf.close()
    
    num_color = 0
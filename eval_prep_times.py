# Measure and evaluate execution times of utilized image processing techniques

import cv2
import numpy as np
import time
from transform import grey_edge
np.set_printoptions(precision=2)
from skimage import io,exposure

# Get reference image for histogram matching
reference = io.imread("320x240_4spec/spec0.jpg") # Change folder name to match create_spectrums.py output

# For each approach, measure 1000 execution times
epochs = ['', '', '', '']
prep_times = []
for i,epoch in enumerate(epochs):
    prep_times.append([])
    source = cv2.VideoCapture('epoch/test/Car_test{}.avi'.format(epoch))
    first = True
    while True:
        ret, frame = source.read()
        if not ret:
            break
        prep = time.time()
        if i == 0:
            frame = cv2.resize(frame, (200,66))
        elif i == 1:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
            frame = cv2.resize(frame, (200,66))
        elif i == 2:
            frame = grey_edge(frame, 4, 2) * 255
            frame = cv2.resize(frame, (200,66))
        elif i == 3:
            frame = cv2.resize(frame, (200,66))
            frame = exposure.match_histograms(frame, reference, multichannel=True)
        frame = np.expand_dims(frame, 0)
        end_prep = time.time() - prep
        if not first:
            prep_times[i].append(end_prep*1000)
        else:
            first = False
    source.release()

# Print the average, standard deviation and maximum value for each approach
for i in range(len(prep_times)):
    print("Approach {}".format(i))
    print("\tAverage: {}".format(np.average(prep_times[i])))
    print("\tStandard Deviation: {}".format(np.std(prep_times[i])))
    print("\tMax Value: {}".format(np.max(prep_times[i])))
    print()
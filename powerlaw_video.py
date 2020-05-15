# Create simulated test data with different lighting conditions by applying Power-Law transformations

import cv2
from skimage import exposure, io
from skimage.exposure import adjust_gamma
try:
    fourcc = cv2.cv.CV_FOURCC(*'XVID')
except AttributeError as e:
    fourcc = cv2.VideoWriter_fourcc(*'XVID')

# Video parameters
num_frames = 1000
fps = 60
res = (320,240)

# Create and save a copy of the transformed test data for each lighting scenario
gammas = [0.33, 4]
for gamma in gammas:
    source = cv2.VideoCapture('epoch/test/Car_test.avi')
    vidfile = cv2.VideoWriter('epoch/test/Car_test_gamma{}.avi'.format(round(gamma)), fourcc, fps, res)
    for i in range(num_frames):
        _,frame = source.read()
        frame = cv2.resize(frame, res)
        frame = adjust_gamma(frame, gamma)
        vidfile.write(frame)
        if i % 100 == 0:
            print("{}".format(i))
    source.release()
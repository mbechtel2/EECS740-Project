# Create simulated training and validation data with different lighting conditions 
# by employing Histogram matching
# Note: this require create_spectrums to have been run first

import cv2
from skimage import exposure, io
from skimage.exposure import match_histograms
from PIL import Image
try:
    fourcc = cv2.cv.CV_FOURCC(*'XVID')
except AttributeError as e:
    fourcc = cv2.VideoWriter_fourcc(*'XVID')

# Video parameters
num_frames = 10000
fps = 60
res = (320,240)

# Create and save a copy of the transformed training data for each lighting scenario
specs = [0,1,2,3]
track_letters = ['i', 'j', 'k', 'l']
for spec in specs:
    reference = io.imread('320x240_4spec/spec{}.jpg'.format(spec))

    source = cv2.VideoCapture('epoch/train/Car_train.avi')
    vidfile = cv2.VideoWriter('epoch/train/Car_train_spec{}.avi'.format(spec), fourcc, fps, res)
    for i in range(num_frames):
        _,frame = source.read()
        frame = cv2.resize(frame, res)
        frame = match_histograms(frame, reference, multichannel=True)
        vidfile.write(frame)
        if i == 0:
            cv2.imwrite("figs/track_spec{}.png".format(spec), frame)
            img = Image.open("figs/track_spec{}.png".format(spec))
            img.save("figs/fig3{}.pdf".format(track_letters[spec]))
        if i % 100 == 0:
            print("{}".format(i))
    source.release()
    
# Create and save a copy of the transformed validation data for each lighting scenario
num_frames = 5000
for spec in specs:
    source = cv2.VideoCapture('epoch/val/Car_val.avi')
    vidfile = cv2.VideoWriter('epoch/val/Car_val_spec{}.avi'.format(spec), fourcc, fps, res)
    for i in range(num_frames):
        _,frame = source.read()
        frame = cv2.resize(frame, res)
        frame = match_histograms(frame, reference, multichannel=True)
        vidfile.write(frame)
        if i % 100 == 0:
            print("{}".format(i))
    source.release()
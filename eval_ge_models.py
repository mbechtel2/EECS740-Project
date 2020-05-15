# Test a CNN model on Grey-Edge Filtered images

import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import cv2
import numpy as np
np.set_printoptions(precision=2)
from transform import grey_edge

# Open the CNN model
model = keras.models.load_model('models/ge-model.h5') 

# Test the model on each lighting scenario and record its outputs
epochs = ['', '_gamma0', '_gamma4'] #, '_dark', '_light']
outputs = []
for i,epoch in enumerate(epochs):
    outputs.append([])
    source = cv2.VideoCapture('epoch/test/Car_test{}.avi'.format(epoch))
    while True:
        ret, frame = source.read()
        if not ret:
            break
        if i == 1:
            frame = grey_edge(frame, 4, 2) * 255
        elif i == 2:
            frame = grey_edge(frame, 4, 3) * 255
        else:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR) 
        frame = cv2.resize(frame, (200,66))
        frame = np.expand_dims(frame, 0)
        outputs[i].append(model.predict(frame)[0][0].item())
    source.release()

# Create a graph of the output sequence on the original test dataset
xvals = range(len(outputs[0]))
plt.figure()
plt.plot(xvals, outputs[0], label='Original')
plt.xlabel("Frame #")
plt.xticks(xvals)
plt.ylabel("Control Output")
plt.ylim((-1,1))
plt.legend()
plt.savefig("figs/fig6a.pdf")

# Create a graph of the output sequence on the dark simulated test dataset
plt.figure()
plt.plot(xvals, outputs[1], label='Dark')
plt.xlabel("Frame #")
plt.xticks(xvals)
plt.ylabel("Control Output")
plt.ylim((-1,1))
plt.legend()
plt.savefig("figs/fig6b.pdf")

# Create a graph of the output sequence on the light simulated test dataset
plt.figure()
plt.plot(xvals, outputs[2], label='Light')
plt.xlabel("Frame #")
plt.xticks(xvals)
plt.ylabel("Control Output")
plt.ylim((-1,1))
plt.legend()
plt.savefig("figs/fig6c.pdf")

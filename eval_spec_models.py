# Test a diversified CNN model on image simulated to represent different lighting conditions

import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import cv2
import numpy as np
np.set_printoptions(precision=2)

# Open the CNN model
model = keras.models.load_model('models/spec-model.h5') 

# Test the model on each lighting scenario and record its outputs
epochs = ['', '_gamma0', '_gamma4']
outputs = []
for i,epoch in enumerate(epochs):
    outputs.append([])
    source = cv2.VideoCapture('epoch/test/Car_test{}.avi'.format(epoch))
    while True:
        ret, frame = source.read()
        if not ret:
            break
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
plt.savefig("figs/fig5a.pdf")

# Create a graph of the output sequence on the dark simulated test dataset
plt.figure()
plt.plot(xvals, outputs[1], label='Dark')
plt.xlabel("Frame #")
plt.xticks(xvals)
plt.ylabel("Control Output")
plt.ylim((-1,1))
plt.legend()
plt.savefig("figs/fig5b.pdf")

# Create a graph of the output sequence on the light simulated test dataset
plt.figure()
plt.plot(xvals, outputs[2], label='Light')
plt.xlabel("Frame #")
plt.xticks(xvals)
plt.ylabel("Control Output")
plt.ylim((-1,1))
plt.legend()
plt.savefig("figs/fig5c.pdf")
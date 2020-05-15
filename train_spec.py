# Train a CNN model using both unmodified track data and track data obtained
# through histogram matching
# Note: this requires match_video.py to have been run first.

import tensorflow as tf
from model import model
import numpy as np
import cv2
from pandas import read_csv

epochs = ['', '_spec0', '_spec1', '_spec2', '_spec3']

# Get training data
train_frames = []
for epoch in epochs:
    vid = cv2.VideoCapture('epoch/train/Car_train{}.avi'.format(epoch))
    while True:
        ret,frame = vid.read()
        if not ret:
            break
        frame = cv2.resize(frame, (200, 66))
        train_frames.append(frame)

epoch = read_csv('epoch/train/Car_train.csv')
vals = epoch.values
train_steers = vals[:,1]
train_steers = np.append(train_steers, vals[:,1])
train_steers = np.append(train_steers, vals[:,1])
train_steers = np.append(train_steers, vals[:,1])
train_steers = np.append(train_steers, vals[:,1])

print(len(train_frames))
print(len(train_steers))

assert len(train_frames) == len(train_steers)

# Get validation data
val_frames = []
for epoch in epochs:
    vid = cv2.VideoCapture('epoch/val/Car_val.avi')
    while True:
        ret,frame = vid.read()
        if not ret:
            break
        frame = cv2.resize(frame, (200, 66))
        val_frames.append(frame)

epoch = read_csv('epoch/val/Car_val.csv')
vals = epoch.values
val_steers = vals[:,1]
val_steers = np.append(val_steers, vals[:,1])
val_steers = np.append(val_steers, vals[:,1])
val_steers = np.append(val_steers, vals[:,1])
val_steers = np.append(val_steers, vals[:,1])
    
print(len(val_frames))
print(len(val_steers))    
    
assert len(val_frames) == len(val_steers)

# Convert the frame data to numpy arrays
train_frames = np.asarray(train_frames)
val_frames = np.asarray(val_frames)

# Train and save the CNN model
model.compile(optimizer=tf.keras.optimizers.Adam(1e-4),
                loss='mse', metrics=['mae'])

history = model.fit(train_frames, train_steers, batch_size=100,
                    epochs=20, validation_data=(val_frames, val_steers))
                    
model.save('models/spec-model.h5') 
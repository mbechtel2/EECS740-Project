# Train a CNN model using unmodified Grayscale track data

import tensorflow as tf
from model import model
import numpy as np
import cv2
from pandas import read_csv

# Get training data
vid = cv2.VideoCapture('epoch/train/Car_train.avi')
train_frames = []
while True:
    ret,frame = vid.read()
    if not ret:
        break
    frame = cv2.resize(frame, (200, 66))
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)    
    train_frames.append(frame)

epoch = read_csv('epoch/train/Car_train.csv')
vals = epoch.values
train_steers = vals[:,1]

assert len(train_frames) == len(train_steers)

# Get validation data
vid = cv2.VideoCapture('epoch/val/Car_val.avi')
val_frames = []
while True:
    ret,frame = vid.read()
    if not ret:
        break
    frame = cv2.resize(frame, (200, 66))
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
    val_frames.append(frame)

epoch = read_csv('epoch/val/Car_val.csv')
vals = epoch.values
val_steers = vals[:,1]
    
assert len(val_frames) == len(val_steers)

# Convert the frame data to numpy arrays
train_frames = np.asarray(train_frames)
val_frames = np.asarray(val_frames)

# Train and save the model
model.compile(optimizer=tf.keras.optimizers.Adam(1e-4),
                loss='mse', metrics=['mae'])

history = model.fit(train_frames, train_steers, batch_size=100,
                    epochs=20, validation_data=(val_frames, val_steers))
                    
model.save('models/ge-model.h5') 
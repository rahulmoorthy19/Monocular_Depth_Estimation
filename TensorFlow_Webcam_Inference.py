import h5py
import time
import numpy as np
import tensorflow as tf
import cv2

## This is a Supporting Class for Custom Layer in Teacher Model - Resnest
'''
ONLY FOR TEACHER MODEL
'''
class rSoftMax(tf.keras.layers.Layer):
    def __init__(self, filters, radix, group_size, **kwargs):
        super(rSoftMax, self).__init__(**kwargs)

        self.filters = filters
        self.radix = radix
        self.group_size = group_size

        if 1 < radix:
            self.seq1 = tf.keras.layers.Reshape([group_size, radix, filters // group_size])
            self.seq2 = tf.keras.layers.Permute([2, 1, 3])
            self.seq3 = tf.keras.layers.Activation(tf.keras.activations.softmax)
            self.seq4 = tf.keras.layers.Reshape([1, 1, radix * filters])
            self.seq = [self.seq1, self.seq2, self.seq3, self.seq4]
        else:
            self.seq1 = tf.keras.layers.Activation(tf.keras.activations.sigmoid)
            self.seq = [self.seq1]

    def call(self, inputs):
        out = inputs
        for l in self.seq:
            out = l(out)
        return out

    def get_config(self):
        config = super(rSoftMax, self).get_config()
        config["filters"] = self.filters
        config["radix"] = self.radix
        config["group_size"] = self.group_size
        return config

MODEL_PATH = "/home/sirzech/Work/UMIN/Vision/Project/Implementation/Rahul_Moorthy_Depth_Estimation/Trained_Models/FastDepth_Model.h5"
## Loading model
model = tf.keras.models.load_model(MODEL_PATH,compile = False)
## Using the Webcam to test the tensorflow model
cap = cv2.VideoCapture(0)
## Setting the Webcam Resolution
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
if not cap.isOpened():
    print("Cannot open camera")
    exit()
new_frame_time = 0
prev_frame_time = 0
while True:
    ret, frame = cap.read()
    new_frame_time = time.time()
    ## Calculating FPS
    fps = 1/(new_frame_time-prev_frame_time)
    prev_frame_time = new_frame_time
    fps = int(fps)
    print("The FPS acheived is:"+str(fps))
    start_time = time.time()
    ## Running inference on image using Tensorflow
    image_array = frame.reshape(1,480, 640, 3)
    depth_output = model.predict(image_array)
    depth_output = depth_output.reshape(480,640)
    end_time = time.time()
    print("Time taken for Inference Using TensorRT:"+str(end_time - start_time))
    ## Showing the Depth Image
    depth_prediction = 255*depth_output
    depth_min = depth_prediction.min()
    depth_max = depth_prediction.max()

    depth_out = (255 * (depth_prediction - depth_min) / (depth_max - depth_min)).astype("uint8")
    cv2.imshow('Webcam Image', frame)
    cv2.imshow('Depth Image', depth_out)
    if cv2.waitKey(1) == ord('q'):
        break

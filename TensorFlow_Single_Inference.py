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
IMAGE_PATH = "/home/sirzech/Work/UMIN/Vision/Project/Implementation/data/nyudepthv2/val/official/00001.h5"
## Loading model
model = tf.keras.models.load_model(MODEL_PATH,compile = False)
## Reading H5 File for Testing Model Inference(VAL Data of NYUv2)
data_input = h5py.File(IMAGE_PATH, 'r')
image_input = data_input.get("rgb")
depth_input = data_input.get("depth")
image_array = np.array(image_input)
depth_array = np.array(depth_input)
image_array = np.moveaxis(image_array, 0, -1)
image_array = image_array.reshape(1,480, 640, 3)
start_time = time.time()
##Prediction on the Image
output = model.predict(image_array)
print("Time taken for Inference by Tensorflow:"+str(time.time() - start_time))
output = output.reshape(480,640)
## Making Depth Image
depth_prediction = 255*output
depth_true = 255*depth_array

depth_min = depth_prediction.min()
depth_max = depth_prediction.max()

real_depth_min = depth_true.min()
real_depth_max = depth_true.max()

img_out = (255 * (depth_prediction - depth_min) / (depth_max - depth_min)).astype("uint8")
img_true = (255 * (depth_true - real_depth_min) / (real_depth_max - real_depth_min)).astype("uint8")
## Saving True Output and Predicted Output
cv2.imwrite("output_tf.png", img_out)
cv2.imwrite("output_true_tf.png", img_true)

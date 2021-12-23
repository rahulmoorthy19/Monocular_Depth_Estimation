import tensorflow as tf
from tensorflow.keras.layers import Conv2D, UpSampling2D, Input, Add
from tensorflow.keras import Model
import numpy as np
from glob import glob
from sklearn.utils import gen_batches
import random
import h5py
import cv2
import warnings
import os
import matplotlib.pyplot as plt

class upsample_block:
    '''
    This is the upsample layer block
    '''
    def __init__(self, input_layer, num_channels):
        self.input_layer = input_layer
        self.num_channels = num_channels
    def create_network(self):
        self.conv_layer = Conv2D(filters=self.num_channels,kernel_size=5,activation='relu',padding='same')(self.input_layer)
        self.upsample_layer = UpSampling2D(size=(2, 2))(self.conv_layer)
        return self.upsample_layer

class upsample_skip_block:
    '''
    This is the upsample skip block which merges the features
    of encoder and upsamples them
    '''
    def __init__(self, input_layer, num_channels, encoder_layer ):
        self.input_layer = input_layer
        self.num_channels = num_channels
        self.encoder_layer = encoder_layer

    def create_network(self):
        self.conv_layer = Conv2D(filters=self.num_channels,kernel_size=5,activation='relu',padding='same')(self.input_layer)
        ## Upsampling
        self.upsample_layer = UpSampling2D(size=(2, 2))(self.conv_layer)
        ## Merging of feature
        self.skip_layer = Add()([self.upsample_layer, self.encoder_layer])
        return self.skip_layer

class Decoder:
    '''
    This is the Decoder architecture
    '''
    def __init__(self, input_layer, encoder_model):
        self.input_layer = encoder_model
        self.input_val = input_layer

    def create_network(self):
        ## Upsample 1
        self.upsample_block_1_obj = upsample_block(self.input_val, 512)
        self.upsample_block_1 = self.upsample_block_1_obj.create_network()
        self.encoder_skip_block_1 = self.input_layer.get_layer('conv_pw_5_relu').output
        ## Upsample Skip 1(Feature fusion)
        self.upsample_block_2_obj = upsample_skip_block(self.upsample_block_1, 256, self.encoder_skip_block_1)
        self.upsample_block_2 = self.upsample_block_2_obj.create_network()
        self.encoder_skip_block_2 = self.input_layer.get_layer('conv_pw_3_relu').output
        ## Upsample Skip 2(Feature fusion)
        self.upsample_block_3_obj = upsample_skip_block(self.upsample_block_2, 128, self.encoder_skip_block_2)
        self.upsample_block_3 = self.upsample_block_3_obj.create_network()
        self.encoder_skip_block_3 = self.input_layer.get_layer('conv_pw_1_relu').output
        ## Upsample Skip 3(Feature fusion)
        self.upsample_block_4_obj = upsample_skip_block(self.upsample_block_3, 64, self.encoder_skip_block_3)
        self.upsample_block_4 = self.upsample_block_4_obj.create_network()
        ## Upsample 2
        self.upsample_block_5_obj = upsample_block(self.upsample_block_4, 32)
        self.upsample_block_5 = self.upsample_block_5_obj.create_network()
        ## Pointwise conv layer for output
        self.output_layer = Conv2D(filters=1,kernel_size=1,padding='same',activation='relu')(self.upsample_block_5)
        return self.output_layer

class FastDepth:
    '''
    This is the main compiling class for fastdepth
    '''
    def __init__(self):
        ## Input
        self.input_layer = tf.keras.layers.Input(shape=(480,640,3))
        ## Resizing for less computation
        self.resize_input = tf.keras.layers.Resizing(224, 224, interpolation='bilinear')(self.input_layer)
        ## Pretrained Mobilenet Architecture(Rich feature extraction)
        self.encoder_model = tf.keras.applications.MobileNet(input_tensor=self.resize_input, alpha=1.0, depth_multiplier=1,
                                                       dropout=0.001,
                                                       include_top=False, weights='imagenet')
        self.encoder_layer = self.encoder_model.get_layer('conv_pw_13_relu').output
        ## Creating the decoder
        self.decoder_obj = Decoder(self.encoder_layer, self.encoder_model)
        self.decoder_output_layer = self.decoder_obj.create_network()
        ## Resizing again to original resolution for loss calculation
        self.resize_output = tf.keras.layers.Resizing(480, 640, interpolation='bilinear')(self.decoder_output_layer)
        ## Model instance
        self.fastdepth_model = Model(inputs= self.input_layer, outputs= self.resize_output)
        self.file_list = list()
        self.train_list = list()
        self.val_list = list()
        self.count = 0
        ## Getting all files path from dataset
        for i in glob("/content/drive/MyDrive/data/*/"):
            for j in glob(i+"*.h5"):
                self.file_list.append(j)
                self.count = self.count + 1
        ## Shuffling the paths
        random.shuffle(self.file_list)
        ## Splitting the data into train and val (Taking part of whole NYU v2 data)
        self.train_list = self.file_list[0:5000]
        self.val_list = self.file_list[5000:6000]
        ## Generating Batches
        self.batches = list(gen_batches(len(self.train_list), 8))
        self.val_batches = list(gen_batches(len(self.val_list), 8))
        random.shuffle(self.batches)
        ## Epoch count
        self.epochs = 15

    def create_batch_generator(self):
        '''
        This is custom batch generator for dealing with large data processing
        '''
        for k in range(self.epochs):
            for count in range(len(self.batches)):
                train_x = list()
                train_y = list()
                for i in range(self.batches[count].start, self.batches[count].stop):
                    data_input = h5py.File(self.train_list[i], 'r')
                    image_input = data_input.get("rgb")
                    depth_input = data_input.get("depth")
                    image_array = np.array(image_input)
                    depth_array = np.array(depth_input)
                    image_array = np.moveaxis(image_array, 0, -1)
                    train_x.append(image_array)
                    train_y.append(depth_array)
                    data_input.close()
                train_x = np.array(train_x)
                train_y = np.array(train_y)
                train_y = train_y.reshape((train_y.shape[0],train_y.shape[1],train_y.shape[2],1))
                yield (train_x,train_y)

    def create_val_batch_generator(self):
        '''
        This is custom batch generator for dealing with large validation data processing
        '''
        for k in range(self.epochs):
            for count in range(len(self.val_batches)):
                train_x = list()
                train_y = list()
                for i in range(self.val_batches[count].start, self.val_batches[count].stop):
                    data_input = h5py.File(self.val_list[i], 'r')
                    image_input = data_input.get("rgb")
                    depth_input = data_input.get("depth")
                    image_array = np.array(image_input)
                    depth_array = np.array(depth_input)
                    image_array = np.moveaxis(image_array, 0, -1)
                    train_x.append(image_array)
                    train_y.append(depth_array)
                    data_input.close()
                train_x = np.array(train_x)
                train_y = np.array(train_y)
                train_y = train_y.reshape((train_y.shape[0],train_y.shape[1],train_y.shape[2],1))
                yield (train_x,train_y)

    def model_return(self):
        '''
        Returns model instance
        '''
        return self.fastdepth_model

    def gradient_matching_loss_func(self, y_true, y_pred):
        '''
        Gradient Matching Loss Implementation
        '''
        resolution = 1
        gradient_matching_loss = 0
        while resolution != 4:
            ##Resizing the image for getting loss at various resolutions
            resized_image_train = tf.image.resize(y_true,
                      (int(y_true.shape[1]/resolution), int(y_true.shape[2]/resolution)))

            resized_image_ground = tf.image.resize(y_pred,
                      (int(y_pred.shape[1]/resolution), int(y_pred.shape[2]/resolution)))
            ## Calculating gradients in x and y direction
            dy_true, dx_true = tf.image.image_gradients(resized_image_train)
            dy_pred, dx_pred = tf.image.image_gradients(resized_image_ground)
            ## Loss Calculation
            final = tf.math.reduce_sum(abs(dy_true*dy_pred) + abs(dx_true*dx_pred))
            gradient_matching_loss += final
            resolution = resolution + 1
        ## normalizing Loss for the batch
        gradient_matching_loss = gradient_matching_loss/(y_true.shape[1]*y_true.shape[2])
        return gradient_matching_loss/y_true.shape[0]

    def scale_invariant_loss_func(self, y_true, y_pred):
        '''
        Scale invariant loss implementation
        '''
        scale_invariant_loss = 0
        ## Calculating log based scale invariant loss
        log_diff = tf.math.log(y_pred) - tf.math.log(y_true)
        num_pixels = tf.cast(y_pred.shape[1]*y_pred.shape[2],tf.float32)
        scale_invariant_loss += tf.math.reduce_sum(tf.math.square(log_diff)) / num_pixels - \
        tf.math.square(tf.math.reduce_sum(log_diff)) / tf.math.square(num_pixels)
        return scale_invariant_loss/y_true.shape[0]

    def depth_loss(self,y_true, y_pred):
        '''
        Calculating Final Loss(Custom Loss Function)
        '''
        gradient_matching_loss = self.gradient_matching_loss_func(y_true, y_pred)
        scale_invariant_loss = self.scale_invariant_loss_func(y_true, y_pred)
        ssim_loss = 1 - tf.reduce_mean(tf.image.ssim(y_true, y_pred, 9.0))
        return gradient_matching_loss + ssim_loss

    def train_model(self):
        '''
        This is the main training pipeline
        '''
        ## Initializing Early Stopping Callback
        callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3)
        ## Compiling the FastDepth Model
        self.fastdepth_model.compile(optimizer=tf.keras.optimizers.Adam(),
                                loss=self.depth_loss,run_eagerly=True,
                                metrics=[tf.keras.metrics.RootMeanSquaredError(),tf.keras.metrics.MeanAbsoluteError()])
        ## Starting the training and storing the loss metrics
        history = self.fastdepth_model.fit(self.create_batch_generator(),validation_data=self.create_val_batch_generator(), steps_per_epoch=len(self.batches),
                                            epochs=self.epochs,validation_steps = len(self.val_batches),
#                                             callbacks=[callback]
                                        )
        ## Plotting the metrics
        plt.plot(history.history['root_mean_squared_error'])
        plt.plot(history.history['val_root_mean_squared_error'])
        plt.title('FastDepth RMSE w/o KD')
        plt.ylabel('RMSE')
        plt.xlabel('epoch')
        plt.legend(['train', 'val'], loc='upper left')
        plt.show()

        plt.plot(history.history['mean_absolute_error'])
        plt.plot(history.history['val_mean_absolute_error'])
        plt.title('FastDepth MAE loss w/o KD')
        plt.ylabel('MAE')
        plt.xlabel('epoch')
        plt.legend(['train', 'val'], loc='upper left')
        plt.show()
        # summarize history for loss
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('FastDepth Loss w/o KD')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'val'], loc='upper left')
        plt.show()
        self.fastdepth_model.save("/content/drive/MyDrive/FastDepth_Model.h5")


## Main Calls
fastdepth = FastDepth()
fastdepth.train_model()

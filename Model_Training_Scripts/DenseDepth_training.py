import tensorflow as tf
from tensorflow.keras.layers import Conv2D, UpSampling2D, Concatenate
from tensorflow.keras import Model
from glob import glob
from sklearn.utils import gen_batches
import h5py
import random
import numpy as np
import cv2
import matplotlib.pyplot as plt

class Upsample_Layer:
    '''
    This is the Upsample Layer of the Decoder
    '''
    def __init__(self,input_layer, encoder_layer, num_channels):
        self.input_layer = input_layer
        self.encoder_layer = encoder_layer
        self.num_channels = num_channels

    def create_network(self):
        self.upsample_layer = UpSampling2D(size=(2, 2))(self.input_layer)
        self.concatenate_layer = Concatenate()([self.upsample_layer,self.encoder_layer])
        self.conv_1 = Conv2D(filters=self.num_channels,kernel_size=3,activation='relu',padding='same')(self.concatenate_layer)
        self.conv_2 = Conv2D(filters=self.num_channels,kernel_size=3,activation='relu',padding='same')(self.conv_1)
        return self.conv_2

class Dense_Depth:
    '''
    This is the DenseDepth Architecture
    '''
    def __init__(self):
        ## Input Layer
        self.input_layer = tf.keras.layers.Input(shape=(480,640,3))
        ## Resizing to 224 x 224 for reducing computation
        self.resize_input = tf.keras.layers.Resizing(224, 224, interpolation='bilinear')(self.input_layer)
        ## Rich Pretrained feature Extractor Encoder
        self.encoder = tf.keras.applications.densenet.DenseNet169(include_top=False, weights='imagenet'
                                                                 ,input_tensor=self.resize_input)
        ## Encoder Output
        self.encoder_output_layer = self.encoder.get_layer('relu').output
        self.decoder_layer_1 = Conv2D(filters=1664,kernel_size=1,activation='relu',padding='same')(self.encoder_output_layer)
        self.encoder_skip_1 = self.encoder.get_layer('pool3_pool').output
        ## Decoder Layer 1
        self.upsample_layer_1_obj = Upsample_Layer(self.decoder_layer_1,self.encoder_skip_1, 832)
        self.upsample_layer_1 = self.upsample_layer_1_obj.create_network()
        self.encoder_skip_2 = self.encoder.get_layer('pool2_pool').output
        ## Decoder Layer 2
        self.upsample_layer_2_obj = Upsample_Layer(self.upsample_layer_1,self.encoder_skip_2, 416)
        self.upsample_layer_2 = self.upsample_layer_2_obj.create_network()
        self.encoder_skip_3 = self.encoder.get_layer('pool1').output
        ## Decoder Layer 3
        self.upsample_layer_3_obj = Upsample_Layer(self.upsample_layer_2,self.encoder_skip_3, 208)
        self.upsample_layer_3 = self.upsample_layer_3_obj.create_network()
        self.encoder_skip_4 = self.encoder.get_layer('conv1/relu').output
        ## Decoder Layer 4
        self.upsample_layer_4_obj = Upsample_Layer(self.upsample_layer_3,self.encoder_skip_4, 104)
        self.upsample_layer_4 = self.upsample_layer_4_obj.create_network()
        ## Decoder Final Layer
        self.final_layer = Conv2D(filters=1,kernel_size=3,activation='relu',padding='same')(self.upsample_layer_4)
        self.final_upsample_layer = UpSampling2D(size=(2, 2))(self.final_layer)
        ## Resizing to original for loss calculation
        self.resize_output = tf.keras.layers.Resizing(480, 640, interpolation='bilinear')(self.final_upsample_layer)
        ## Creation of Model Instance
        self.densedepth_model = Model(inputs= self.input_layer, outputs= self.resize_output)
        self.file_list = list()
        self.train_list = list()
        self.val_list = list()
        self.count = 0
        ## Getting all files path from dataset
        for i in glob("/home/sirzech/Work/UMIN/Vision/Project/Implementation/data/nyudepthv2/data/*/"):
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

    def model_return(self):
        '''
        This function return the model Instance
        '''
        return self.densedepth_model

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
        ## Structural loss for brining in similarity between the structure
        ssim_loss = 1 - tf.reduce_mean(tf.image.ssim(y_true, y_pred, 9.0))
        return gradient_matching_loss + ssim_loss

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

    def train_model(self):
        '''
        This is the main training pipeline
        '''
        ## Initializing Early Stopping Callback
        callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=3)
        ## Compiling the DenseDepth Model
        self.densedepth_model.compile(optimizer=tf.keras.optimizers.Adam(),
                                loss=self.depth_loss,run_eagerly=True,
                                metrics=[tf.keras.metrics.RootMeanSquaredError(),tf.keras.metrics.MeanAbsoluteError()])
        ## Starting the training and storing the loss metrics
        history = self.densedepth_model.fit(self.create_batch_generator(),validation_data=self.create_val_batch_generator(), steps_per_epoch=len(self.batches),
                                            epochs=self.epochs,validation_steps = len(self.val_batches),
#                                             callbacks=[callback]
                                        )
        ## Plotting the metrics
        plt.plot(history.history['root_mean_squared_error'])
        plt.plot(history.history['val_root_mean_squared_error'])
        plt.title('Teacher Model RMSE')
        plt.ylabel('RMSE')
        plt.xlabel('epoch')
        plt.legend(['train', 'val'], loc='upper left')
        plt.show()

        plt.plot(history.history['mean_absolute_error'])
        plt.plot(history.history['val_mean_absolute_error'])
        plt.title('Teacher Model MAE loss')
        plt.ylabel('MAE')
        plt.xlabel('epoch')
        plt.legend(['train', 'val'], loc='upper left')
        plt.show()
        # summarize history for loss
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('DenseDepth Model Loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'val'], loc='upper left')
        plt.show()
        self.densedepth_model.save("DenseDepth_Model.h5")

## Main Calls
model = Dense_Depth()
model.train_model()

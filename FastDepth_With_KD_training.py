import tensorflow as tf
from tensorflow.keras.layers import Conv2D, UpSampling2D, Input, Add
from tensorflow.keras import Model
import numpy as np
import h5py
import os
import cv2
from sklearn.utils import gen_batches
from glob import glob
import random
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

class upsample_block:
    '''
    This is the Upsample block
    '''
    def __init__(self, input_layer, num_channels,name):
        self.input_layer = input_layer
        self.num_channels = num_channels
        self.name = name
    def create_network(self):
        '''
        This is the main Upsample network
        '''
        self.conv_layer = Conv2D(filters=self.num_channels,kernel_size=5,activation='relu',padding='same')(self.input_layer)
        self.upsample_layer = UpSampling2D(size=(2, 2),name=self.name)(self.conv_layer)
        return self.upsample_layer

class upsample_skip_block:
    '''
    This is the upsample skip block which does feature fusion
    '''
    def __init__(self, input_layer, num_channels, encoder_layer,name):
        self.input_layer = input_layer
        self.num_channels = num_channels
        self.encoder_layer = encoder_layer
        self.name = name

    def create_network(self):
        '''
        This is the feature fusion network
        '''
        self.conv_layer = Conv2D(filters=self.num_channels,kernel_size=5,activation='relu',padding='same')(self.input_layer)
        self.upsample_layer = UpSampling2D(size=(2, 2))(self.conv_layer)
        self.skip_layer = Add(name=self.name)([self.upsample_layer, self.encoder_layer])
        return self.skip_layer

class Student_Decoder:
    '''
    This is the Student Decoder network
    '''
    def __init__(self, input_layer, encoder_model):
        self.input_layer = encoder_model
        self.input_val = input_layer

    def create_network(self):
        '''
        This is the Student decoder network
        '''
        ## Upsample 1
        self.upsample_block_1_obj = upsample_block(self.input_val, 512,name = "Upsample_1")
        self.upsample_block_1 = self.upsample_block_1_obj.create_network()
        ## Upsample skip 1
        self.encoder_skip_block_1 = self.input_layer.get_layer('conv_pw_5_relu').output
        self.upsample_block_2_obj = upsample_skip_block(self.upsample_block_1, 256, self.encoder_skip_block_1,name="Upsample_skip_1")
        self.upsample_block_2 = self.upsample_block_2_obj.create_network()
        ## Upsample skip 2
        self.encoder_skip_block_2 = self.input_layer.get_layer('conv_pw_3_relu').output
        self.upsample_block_3_obj = upsample_skip_block(self.upsample_block_2, 128, self.encoder_skip_block_2,name="Upsample_skip_2")
        self.upsample_block_3 = self.upsample_block_3_obj.create_network()
        ## Upsample skip 3
        self.encoder_skip_block_3 = self.input_layer.get_layer('conv_pw_1_relu').output
        self.upsample_block_4_obj = upsample_skip_block(self.upsample_block_3, 64, self.encoder_skip_block_3,name="Upsample_skip_3")
        self.upsample_block_4 = self.upsample_block_4_obj.create_network()
        ## Upsample 2
        self.upsample_block_5_obj = upsample_block(self.upsample_block_4, 32,name = "Upsample_2")
        self.upsample_block_5 = self.upsample_block_5_obj.create_network()
        ## Pointwise Layer
        self.output_layer = Conv2D(filters=1,kernel_size=1,padding='same',activation='relu',name="Pointwise_Layer")(self.upsample_block_5)
        return self.output_layer

class rSoftMax(tf.keras.layers.Layer):
    '''
    This is for loading the Teacher Model
    '''
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

class Student_Network:
    '''
    This is the student network main class
    '''
    def __init__(self):
        ## Input Layer
        self.input_layer = tf.keras.layers.Input(shape=(480,640,3))
        ## This is for resizing the input for reducing the computation
        self.resize_input = tf.keras.layers.Resizing(224, 224, interpolation='bilinear')(self.input_layer)
        ## This is the Pretrained encoder model for feature rich extraction
        self.encoder_model = tf.keras.applications.MobileNet(input_tensor=self.resize_input, alpha=1.0, depth_multiplier=1,
                                                       dropout=0.001,
                                                       include_top=False, weights='imagenet')
        self.encoder_layer = self.encoder_model.get_layer('conv_pw_13_relu').output
        ## Creating Student Decoder
        self.decoder_obj = Student_Decoder(self.encoder_layer, self.encoder_model)
        self.decoder_output_layer = self.decoder_obj.create_network()
        ## Resizing the layer for calculating the loss
        self.resize_output = tf.keras.layers.Resizing(480, 640, interpolation='bilinear')(self.decoder_output_layer)
        self.student_model = Model(inputs= self.encoder_model.input, outputs= self.resize_output)
        ## Loading the teacher network
        self.teacher_model = tf.keras.models.load_model("Teacher_Model.h5",compile=False,
                                                custom_objects = {"rSoftMax":rSoftMax})
        ## Extracting the intermediate layers of the teacher network for KD Calculation
        self.teacher_model_intermediate_1 = Model(inputs=self.teacher_model.input,
                                              outputs=self.teacher_model.get_layer("Add_2").output)
        self.teacher_model_intermediate_2 = Model(inputs=self.teacher_model.input,
                                              outputs=self.teacher_model.get_layer("Add_3").output)
        ##Dataset Preparation
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
        self.batches = list(gen_batches(len(self.train_list), 4))
        self.val_batches = list(gen_batches(len(self.val_list), 4))
        random.shuffle(self.batches)
        ## Epoch count
        self.epochs = 15

    def MAE(self, y_ground, y_pred):
        '''
        Calculation Mean absolute error metric
        '''
        y_true = y_ground[:,:,:,0]
        y_true = tf.reshape(y_true,[y_true.shape[0],y_true.shape[1], y_true.shape[2], 1])
        return tf.keras.metrics.mean_absolute_error(y_true, y_pred)


    def RMSE(self, y_ground, y_pred):
        '''
        Calculation Root Mean Square error metric
        '''
        y_true = y_ground[:,:,:,0]
        y_true = tf.reshape(y_true,[y_true.shape[0],y_true.shape[1], y_true.shape[2], 1])
        return tf.math.sqrt(tf.math.reduce_mean(tf.math.square(y_pred - y_true)))

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
                    depth_array = depth_array.reshape((depth_array.shape[0],depth_array.shape[1],1))
                    depth_array = np.concatenate((depth_array, image_array), axis=-1)
                    train_x.append(image_array)
                    train_y.append(depth_array)
                    data_input.close()
                train_x = np.array(train_x)
                train_y = np.array(train_y)
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
                    depth_array = depth_array.reshape((depth_array.shape[0],depth_array.shape[1],1))
                    depth_array = np.concatenate((depth_array, image_array), axis=-1)
                    train_x.append(image_array)
                    train_y.append(depth_array)
                    data_input.close()
                train_x = np.array(train_x)
                train_y = np.array(train_y)
                yield (train_x,train_y)


    def model_return(self):
        '''
        This function return the model Instance
        '''
        return self.student_model

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


    def pairwise_KD_loss(self,x_data, beta):
        '''
        This is function which calculates Knowledge Distillation Loss
        '''
        pairwise_knowledge_distillation = 0
        ## Extracting Student Intermediate model for calculating KD loss
        student_model_intermediate_1 = Model(inputs=self.student_model.input,
                                              outputs=self.student_model.get_layer("Upsample_skip_1").output)
        student_model_intermediate_2 = Model(inputs=self.student_model.input,
                                              outputs=self.student_model.get_layer("Upsample_skip_2").output)

        ## Extracting same spatial intermidiate layers
        teacher_predictions_1 = self.teacher_model_intermediate_1(x_data)
        teacher_predictions_2 = self.teacher_model_intermediate_2(x_data)
        student_predictions_1 = student_model_intermediate_1(x_data)
        student_predictions_2 = student_model_intermediate_2(x_data)

        ## Average Pooling
        nodes_teacher_1 = tf.keras.layers.AveragePooling2D(pool_size=(beta, beta),
                          strides=(beta, beta), padding='valid')(teacher_predictions_1)
        nodes_teacher_2 = tf.keras.layers.AveragePooling2D(pool_size=(beta, beta),
                          strides=(beta, beta), padding='valid')(teacher_predictions_2)
        nodes_student_1 = tf.keras.layers.AveragePooling2D(pool_size=(beta, beta),
                          strides=(beta, beta), padding='valid')(student_predictions_1)
        nodes_student_2 = tf.keras.layers.AveragePooling2D(pool_size=(beta, beta),
                          strides=(beta, beta), padding='valid')(student_predictions_2)

        ## Calculating Affinity maps
        nodes_teacher_1 = tf.transpose(nodes_teacher_1, perm=[0,3,1,2])
        nodes_teacher_2 = tf.transpose(nodes_teacher_2, perm=[0,3,1,2])
        nodes_student_1 = tf.transpose(nodes_student_1, perm=[0,3,1,2])
        nodes_student_2 = tf.transpose(nodes_student_2, perm=[0,3,1,2])

        nodes_teacher_1 = tf.reshape(nodes_teacher_1,[nodes_teacher_1.shape[0],
                                                   nodes_teacher_1.shape[1],
                                                   nodes_teacher_1.shape[2] * nodes_teacher_1.shape[3]])



        nodes_teacher_2 = tf.reshape(nodes_teacher_2,[nodes_teacher_2.shape[0],
                                                   nodes_teacher_2.shape[1],
                                                   nodes_teacher_2.shape[2]* nodes_teacher_2.shape[3]])



        nodes_student_1 = tf.reshape(nodes_student_1,[nodes_student_1.shape[0],
                                                   nodes_student_1.shape[1],
                                                   nodes_student_1.shape[2]* nodes_student_1.shape[3]])


        nodes_student_2 = tf.reshape(nodes_student_2,[nodes_student_2.shape[0],
                                                   nodes_student_2.shape[1],
                                                   nodes_student_2.shape[2] * nodes_student_2.shape[3]])

        ## Calculating Knowledge distillation
        for i in range(nodes_student_2.shape[0]):
            t_1 = tf.linalg.matmul(tf.transpose(nodes_teacher_1[i,:,:]),nodes_teacher_1[i,:,:])
            t_2 = tf.linalg.matmul(tf.transpose(nodes_teacher_2[i,:,:]),nodes_teacher_2[i,:,:])

            s_1 = tf.linalg.matmul(tf.transpose(nodes_student_1[i,:,:]),nodes_student_1[i,:,:])
            s_2 = tf.linalg.matmul(tf.transpose(nodes_student_2[i,:,:]),nodes_student_2[i,:,:])

            teacher_norm_1 = tf.norm(nodes_teacher_1[i,:,:],axis = 0)
            teacher_norm_1 = tf.reshape(teacher_norm_1,[1,teacher_norm_1.shape[0]])
            teacher_norm_2 = tf.norm(nodes_teacher_2[i,:,:],axis = 0)
            teacher_norm_2 = tf.reshape(teacher_norm_2,[1,teacher_norm_2.shape[0]])

            student_norm_1 = tf.norm(nodes_student_1[i,:,:],axis = 0)
            student_norm_1 = tf.reshape(student_norm_1, [1,student_norm_1.shape[0]])
            student_norm_2 = tf.norm(nodes_student_2[i,:,:],axis = 0)
            student_norm_2 = tf.reshape(student_norm_2, [1,student_norm_2.shape[0]])

            t_1_norm = tf.linalg.matmul(tf.transpose(teacher_norm_1),teacher_norm_1)
            t_2_norm = tf.linalg.matmul(tf.transpose(teacher_norm_2),teacher_norm_2)
            s_1_norm = tf.linalg.matmul(tf.transpose(student_norm_1),student_norm_1)
            s_2_norm = tf.linalg.matmul(tf.transpose(student_norm_2),student_norm_2)

            te_1 = tf.divide(t_1,t_1_norm)
            te_2 = tf.divide(t_2,t_2_norm)
            se_1 = tf.divide(s_1,s_1_norm)
            se_2 = tf.divide(s_2,s_2_norm)
            Beta = beta*beta
            Alpha = (teacher_predictions_1.shape[1]/beta)*(teacher_predictions_1.shape[2]/beta)
            Alpha_1 = (teacher_predictions_2.shape[1]/beta)*(teacher_predictions_2.shape[2]/beta)
            pairwise_knowledge_distillation += ((tf.math.reduce_sum(tf.math.square(se_1 - te_1))*Beta)/(Alpha)) +\
                        ((tf.math.reduce_sum(tf.math.square(se_2 - te_2))*Beta)/(Alpha_1))
        ## Deleting model instance for saving memory
        del student_model_intermediate_1
        del student_model_intermediate_2
        return  pairwise_knowledge_distillation/nodes_teacher_1.shape[0]

    def student_loss(self,y_ground, y_pred):
        '''
        This is the final student loss
        '''
        y_true = y_ground[:,:,:,0]
        y_true = tf.reshape(y_true,[y_true.shape[0],y_true.shape[1], y_true.shape[2], 1])
        x_data = y_ground[:,:,:,1:]
        gradient_matching_loss = self.gradient_matching_loss_func(y_true, y_pred)
        ssim_loss = 1 - tf.reduce_mean(tf.image.ssim(y_true, y_pred, 9.0))
        #scale_invariant_loss = self.scale_invariant_loss_func(y_true, y_pred)
        final_pairwise_kd_loss = self.pairwise_KD_loss(x_data,2)
        return gradient_matching_loss + ssim_loss + final_pairwise_kd_loss

    def train_model(self):
        '''
        This is the main training pipeline
        '''
        ## Initializing Early Stopping Callback
        callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=3)
        ## Compiling the Student Model
        self.student_model.compile(optimizer=tf.keras.optimizers.Adam(),
                                loss=self.student_loss,run_eagerly=True,
                                metrics=[self.RMSE,self.MAE])
        ## Starting the training and storing the loss metrics
        history = self.student_model.fit(self.create_batch_generator(),validation_data=self.create_val_batch_generator(), steps_per_epoch=len(self.batches),
                                            epochs=self.epochs,validation_steps = len(self.val_batches),
#                                             callbacks=[callback]
                                        )
        ## Plotting the metrics
        plt.plot(history.history['RMSE'])
        plt.plot(history.history['val_RMSE'])
        plt.title('FastDepth with KD RMSE')
        plt.ylabel('RMSE')
        plt.xlabel('epoch')
        plt.legend(['train', 'val'], loc='upper left')
        plt.show()

        plt.plot(history.history['MAE'])
        plt.plot(history.history['val_MAE'])
        plt.title('FastDepth with KD MAE loss')
        plt.ylabel('MAE')
        plt.xlabel('epoch')
        plt.legend(['train', 'val'], loc='upper left')
        plt.show()
        # summarize history for loss
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('FastDepth with KD Loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'val'], loc='upper left')
        plt.show()
        self.student_model.save("FastDepth_KD_Model.h5")


## Main Calls
student_model = Student_Network()
final_student_model = student_model.train_model()

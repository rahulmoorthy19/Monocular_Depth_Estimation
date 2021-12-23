from collections import namedtuple
import json
import numpy as np
import pycuda.driver as cuda
import pycuda.autoinit
from pathlib import Path
import tensorflow as tf
import tensorrt as trt
import cv2
import time
## Please change this path for your model
ONNX_PATH = "/home/sirzech/Work/UMIN/Vision/Project/Implementation/teacher_model.onnx"
HostDeviceMemory = namedtuple('HostDeviceMemory', 'host_memory device_memory')
TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

EXPLICIT_BATCH = 1 << (int)(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
def GiB(val):
    return val * 1 << 30

class TensorRTDepthEstimation:
    '''
    Uses TensorRT to do the inference.
    '''
    def __init__(self):
        '''
        This Function Initializes the buffers
        '''
        self.logger = trt.Logger(trt.Logger.WARNING)
        self.engine = None
        self.inputs = None
        self.outputs = None
        self.bindings = None
        self.stream = None

    def allocate_buffers(self, engine):
        '''
        This function allocates the buffers for prediction
        '''
        inputs = []
        outputs = []
        bindings = []
        stream = cuda.Stream()
        for binding in engine:
            size = trt.volume(engine.get_binding_shape(binding)) * engine.max_batch_size
            dtype = trt.nptype(engine.get_binding_dtype(binding))
            # Allocate host and device buffers
            host_memory = cuda.pagelocked_empty(size, dtype)
            device_memory = cuda.mem_alloc(host_memory.nbytes)
            bindings.append(int(device_memory))
            if engine.binding_is_input(binding):
                inputs.append(HostDeviceMemory(host_memory, device_memory))
            else:
                outputs.append(HostDeviceMemory(host_memory, device_memory))

        return inputs, outputs, bindings, stream

    def build_engine_onnx(self, model_file):
        '''
        This Function is used to build the TensorRT Engine
        '''
        builder = trt.Builder(TRT_LOGGER)
        network = builder.create_network(EXPLICIT_BATCH)
        config = builder.create_builder_config()
        parser = trt.OnnxParser(network, TRT_LOGGER)
        config.max_workspace_size = GiB(1)
        ## Setting Profile shapes for dynamic Optmization
        profile = builder.create_optimization_profile()
        profile.set_shape("input_1", (1,480,640,3),(1,480,640,3),(1,480,640,3))
        config.add_optimization_profile(profile)
        # Load the Onnx model and parse it in order to populate the TensorRT network.
        with open(model_file, 'rb') as model:
            if not parser.parse(model.read()):
                print ('ERROR: Failed to parse the ONNX file.')
                for error in range(parser.num_errors):
                    print (parser.get_error(error))
                return None
        ## Creating TensorRT Engine
        self.engine = builder.build_engine(network, config)
        print('Allocating Buffers')
        ## Allocating Buffers for Prediction
        self.inputs, self.outputs, self.bindings, self.stream = self.allocate_buffers(self.engine)
        print('Ready')

    def infer(self, context, bindings, inputs, outputs, stream, batch_size=1):
        '''
        This Function consists of the inference cycle
        '''
        # Transfer input data to the GPU.
        [cuda.memcpy_htod_async(inp.device_memory, inp.host_memory, stream) for inp in inputs]
        # Run inference.
        context.execute_async(batch_size=batch_size, bindings=bindings, stream_handle=stream.handle)
        # Transfer predictions back from the GPU.
        [cuda.memcpy_dtoh_async(out.host_memory, out.device_memory, stream) for out in outputs]
        # Synchronize the stream
        stream.synchronize()
        # Return only the host outputs.
        return [out.host_memory for out in outputs]

    def run(self, image):
        '''
        This is the compiling function which combines the above functions
        for inference cycle
        '''
        # Flatten it to a 1D array.
        image = image.ravel()
        # The first input is the image. Copy to host memory.
        image_input = self.inputs[0]
        np.copyto(image_input.host_memory, image)
        with self.engine.create_execution_context() as context:
            ## Getting Output from TensorRT engine
            output = self.infer(context=context,bindings=self.bindings,
                                                    inputs=self.inputs,
                                                    outputs=self.outputs,
                                                    stream=self.stream)
            ## Reshaping for Model Output
            return output[0].reshape(480,640)

## Creating TensorRT object
tensorrt_model = TensorRTDepthEstimation()
## Creating TensorRT Engine from ONNX model

tensorrt_model.build_engine_onnx(ONNX_PATH)
## Using the Webcam to test the TensorRT model
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
    ## Running inference on image using TensorRT
    depth_output = tensorrt_model.run(frame)
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

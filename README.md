# Monocular Depth Estimation

The project aimed to develop a model for monocular Depth Estimation which runs real time on an embedded device(Jetson Nano).  In particular, a small extension of the FastDepth model was developed which performs better than the original model.  The contributions and novelty achieved through the project is as follows -



•Extending  the  original  Fast  Depth  Model  work  using  Knowledge  Distillation  Loss  based Teacher-Student technique for improving the performance.


•Deploying the Depth Estimation model using TensorRT rather than Apache TVM which theoriginal paper proposed for studying modern edge device deploying techniques

import tensorflow as tf

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

## Converting Teacher Model to Tensorflow Saved Model
# model = tf.keras.models.load_model("Teacher_Model.h5",custom_objects = {"rSoftMax":rSoftMax},compile = False)
# tf.saved_model.save(model, "teacher_model")

## Converting FastDepth KD Model to Tensorflow Saved Model
# model = tf.keras.models.load_model("FastDepth_KD_Model.h5",compile = False)
# tf.saved_model.save(model, "FastDepth_KD")

## Converting FastDepth without KD Model to Tensorflow Saved Model
# model = tf.keras.models.load_model("FastDepth.h5",compile = False)
# tf.saved_model.save(model, "FastDepth")

## Converting DenseDepth Model to Tensorflow Saved Model
# model = tf.keras.models.load_model("DenseDepth.h5",compile = False)
# tf.saved_model.save(model, "DenseDepth")

## This is the sample command for converting saved model to onnx model
#python -m tf2onnx.convert --saved-model teacher_model --output teacher_model.onnx

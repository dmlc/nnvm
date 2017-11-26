import numpy as np
import nnvm
import topi
import tvm
from tvm.contrib import graph_runtime
from nnvm.testing.config import ctx_list
import keras
import model_zoo

# prevent keras from using all gpu memory
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.4
set_session(tf.Session(config=config))


def verify_keras_frontend(keras_model, input_shape=(1,3,224,224), out_shape=(1,1000)):
    def get_keras_output(x, dtype='float32'):
        return keras_model.predict(x.transpose([0,2,3,1]))

    def get_tvm_output(x, target, ctx, input_name='data', dtype='float32'):
        sym, params = nnvm.frontend.from_keras(keras_model)
        shape_dict = {input_name : x.shape}
        with nnvm.compiler.build_config(opt_level=3):
            graph, lib, params = nnvm.compiler.build(sym, target, shape_dict, params=params)
        m = graph_runtime.create(graph, lib, ctx)
        m.set_input(input_name, tvm.nd.array(x.astype(dtype)))
        m.set_input(**params)
        m.run()
        out = m.get_output(0, tvm.nd.empty(out_shape, dtype))
        return out.asnumpy()

    x = np.random.uniform(size=input_shape)
    keras_out = get_keras_output(x)
    for target, ctx in ctx_list():
        tvm_out = get_tvm_output(x, target, ctx)
        np.testing.assert_allclose(keras_out, tvm_out, rtol=1e-5, atol=1e-5)


if __name__ == '__main__':
    # verify_keras_frontend(model_zoo.get_mlp(), (1,3,224,224))
    verify_keras_frontend(model_zoo.get_vgg16(), (1,3,224,224))
    # verify_keras_frontend(model_zoo.get_xception(), (1,3,299,299))
    # verify_keras_frontend(model_zoo.get_mobilenet(), (1,3,224,224))
    # verify_keras_frontend(model_zoo.get_resnet50(), (1,3,224,224))

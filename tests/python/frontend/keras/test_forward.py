import numpy as np
import nnvm
import tvm
from tvm.contrib import graph_runtime
from nnvm.testing.config import ctx_list
import keras

# prevent keras from using up all gpu memory
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.5
set_session(tf.Session(config=config))


def verify_keras_frontend(keras_model):
    in_shape = [dim.value if dim.value is not None else 1 for dim in keras_model.input_layers[0].input.shape]
    out_shape = [dim.value if dim.value is not None else 1 for dim in keras_model.get_output_at(0).shape]

    def replace_nones(shape):
        return list([s if s else 128 for s in shape])

    def get_keras_output(x, dtype='float32'):
        return keras_model.predict(x)

    def get_tvm_output(x, target, ctx, input_name='data', dtype='float32'):
        sym, params = nnvm.frontend.from_keras(keras_model)
        shape_dict = {input_name : x.shape}
        with nnvm.compiler.build_config(opt_level=2):
            graph, lib, params = nnvm.compiler.build(sym, target, shape_dict, params=params)
        m = graph_runtime.create(graph, lib, ctx)
        m.set_input(input_name, tvm.nd.array(x.astype(dtype)))
        m.set_input(**params)
        m.run()
        out = m.get_output(0, tvm.nd.empty(out_shape, dtype))
        return out.asnumpy()

    x = np.random.uniform(size=replace_nones(in_shape))
    keras_out = get_keras_output(x)
    for target, ctx in ctx_list():
        tvm_out = get_tvm_output(x.transpose([0,3,1,2]), target, ctx)
        np.testing.assert_allclose(keras_out, tvm_out, rtol=1e-5, atol=1e-5)


def test_forward_softrelu():
    print("test_forward_softrelu")
    data = keras.layers.Input(shape=(32,32,3))
    x = keras.layers.Activation('softplus')(data)
    x = keras.layers.Concatenate()([x, x])
    x = keras.layers.GlobalMaxPooling2D()(x)
    keras_model = keras.models.Model(data, x)
    verify_keras_frontend(keras_model)


def test_forward_leaky_relu():
    print("test_forward_leaky_relu")
    data = keras.layers.Input(shape=(32,32,3))
    x = keras.layers.LeakyReLU(alpha=0.3)(data)
    x = keras.layers.Add()([x, x])
    x = keras.layers.GlobalAveragePooling2D()(x)
    keras_model = keras.models.Model(data, x)
    verify_keras_frontend(keras_model)


def test_forward_multiply():
    print("test_forward_multiply")
    data =  keras.layers.Input(shape=(32,32,3))
    x = keras.layers.GlobalAveragePooling2D()(data)
    x = keras.layers.Reshape((1, 1, 3))(x)
    x = keras.layers.multiply([x, data])
    x = keras.layers.GlobalAveragePooling2D()(x)
    keras_model = keras.models.Model(data, x)
    verify_keras_frontend(keras_model)


def test_forward_dense():
    print("test_forward_dense")
    data = keras.layers.Input(shape=(32,32,3))
    x = keras.layers.MaxPooling2D(pool_size=(2,2))(data)
    x = keras.layers.Flatten()(x)
    x = keras.layers.Dense(10, activation='relu', kernel_initializer='uniform')(x)
    keras_model = keras.models.Model(data, x)
    verify_keras_frontend(keras_model)


def test_forward_conv():
    print("test_forward_conv")
    shape = [15, 16]
    kernel = range(1, 8)
    stride = [1, 2, 3]
    for i in shape:
        for k in kernel:
            for s in stride:
                print(i,k,s)
                data = keras.layers.Input(shape=(i,i,3))
                x = keras.layers.Conv2D(filters=10, kernel_size=(k,k), strides=(s,s), padding='same')(data)
                x = keras.layers.GlobalMaxPooling2D()(x)
                keras_model = keras.models.Model(data, x)
                verify_keras_frontend(keras_model)


def test_forward_conv_small():
    print("test_forward_conv_small")
    shape = [4, 5]
    kernel = range(1, 8)
    stride = [1, 2]
    for i in shape:
        for k in kernel:
            for s in stride:
                print(i,k,s)
                data = keras.layers.Input(shape=(i,i,3))
                x = keras.layers.Conv2D(filters=10, kernel_size=(k,k), strides=(s,s), padding='same')(data)
                x = keras.layers.GlobalMaxPooling2D()(x)
                keras_model = keras.models.Model(data, x)
                verify_keras_frontend(keras_model)


def test_forward_transpose_conv():
    print("test_forward_transpose_conv")
    shape = [15, 16]
    kernel = range(1, 8)
    stride =[1, 2]
    for i in shape:
        for k in kernel:
            for s in stride:
                print(i,k,s)
                data = keras.layers.Input(shape=(i,i,3))
                x = keras.layers.Conv2DTranspose(filters=10, kernel_size=(k,k), strides=(s,s), padding='same')(data)
                x = keras.layers.GlobalAveragePooling2D()(x)
                keras_model = keras.models.Model(data, x)
                verify_keras_frontend(keras_model)


def test_forward_depthwise_conv():
    print("test_forward_depthwise_conv")
    shape = [15, 16]
    kernel = range(1, 8)
    stride = [1, 2, 3]
    for i in shape:
        for k in kernel:
            for s in stride:
                print(i,k,s)
                data = keras.layers.Input(shape=(i,i,3))
                x = keras.applications.mobilenet.DepthwiseConv2D(kernel_size=(k,k), strides=(s,s), padding='same')(data)
                x = keras.layers.GlobalAveragePooling2D()(x)
                keras_model = keras.models.Model(data, x)
                verify_keras_frontend(keras_model)


def test_forward_separable_conv():
    print("test_forward_separable_conv")
    shape = [15, 16]
    kernel = range(1, 8)
    stride = [1, 2, 3]
    for i in shape:
        for k in kernel:
            for s in stride:
                print(i,k,s)
                data = keras.layers.Input(shape=(i,i,3))
                x = keras.layers.SeparableConv2D(filters=10, kernel_size=(k,k), strides=(s,s),
                    padding='same', activation='relu')(data)
                x = keras.layers.BatchNormalization(scale=True, center=False,
                    beta_initializer='uniform', gamma_initializer='uniform')(x)
                x = keras.layers.GlobalAveragePooling2D()(x)
                keras_model = keras.models.Model(data, x)
                verify_keras_frontend(keras_model)


def test_forward_upsample():
    print("test_forward_upsample")
    shape = [15, 16]
    pool = range(1, 10)
    for i in shape:
        for p in pool:
            print(i, p)
            data = keras.layers.Input(shape=(i,i,3))
            x = keras.layers.UpSampling2D(size=(p,p))(data)
            x = keras.layers.GlobalAveragePooling2D()(x)
            keras_model = keras.models.Model(data, x)
            verify_keras_frontend(keras_model)


def test_forward_pooling():
    print("test_forward_pooling")
    shape = [15, 16]
    pool = [1, 2, 3, 4]
    stride = [1, 2, 3, 4]
    for i in shape:
        for p in pool:
            for s in stride:
                print(i, p, s)
                data = keras.layers.Input(shape=(i,i,3))
                x = keras.layers.MaxPooling2D(pool_size=(p,p), strides=(s,s), padding="same")(data)
                x = keras.layers.GlobalAveragePooling2D()(x)
                keras_model = keras.models.Model(data, x)
                verify_keras_frontend(keras_model)


def test_forward_seblock():
    print("test_forward_reshape")
    filters = 16
    data = keras.layers.Input(shape=(128,128,3))
    x = keras.layers.Conv2D(filters, (3, 3), padding='same')(data)

    # squeeze-excite block formulation
    r = keras.layers.GlobalAveragePooling2D()(data)
    r = keras.layers.Reshape((1, 1, -1))(r)
    r = keras.layers.Dense(filters // 4, activation='relu', kernel_initializer='he_normal', use_bias=False)(r)
    r = keras.layers.Dense(filters, activation='sigmoid', kernel_initializer='he_normal', use_bias=False)(r)
    x = keras.layers.multiply([x, r])

    x = keras.layers.GlobalAveragePooling2D()(x)
    keras_model = keras.models.Model(data, x)
    verify_keras_frontend(keras_model)


def test_forward_shape_inference():
    print("test_forward_shape_inference")
    data = keras.layers.Input(shape=(None, None, 3))
    x = keras.layers.Conv2D(filters=10, kernel_size=(3, 3), padding='same')(data)
    x = keras.layers.AveragePooling2D()(x)
    x = keras.layers.UpSampling2D()(x)
    x = keras.layers.Conv2D(filters=3, kernel_size=(3, 3), padding='same')(data)
    x = keras.layers.GlobalAveragePooling2D(x)
    keras_model = keras.models.Model(data, x)
    verify_keras_frontend(keras_model)


def test_forward_vgg16():
    print("test_forward_vgg16")
    keras_model = keras.applications.vgg16.VGG16(include_top=True, weights=None,
        input_shape=(224,224,3), classes=1000)
    verify_keras_frontend(keras_model)


def test_forward_xception():
    print("test_forward_xception")
    keras_model = keras.applications.xception.Xception(include_top=True, weights=None,
        input_shape=(299,299,3), classes=1000)
    verify_keras_frontend(keras_model)


def test_forward_resnet50():
    print("test_forward_resnet50")
    keras_model = keras.applications.resnet50.ResNet50(include_top=True, weights=None,
        input_shape=(224,224,3), classes=1000)
    verify_keras_frontend(keras_model)


def test_forward_unet():
    print("test_forward_unet")
    skip = []
    x = data = keras.layers.Input(shape=(256, 256, 3))
    for i in range(4):
        x = keras.layers.Conv2D(filters=8*(2**i), kernel_size=(3, 3), padding='same', activation='relu')(x)
        x = keras.layers.MaxPooling2D()(x)
        skip.append(x)
    for i in range(3, -1, -1):
        x = keras.layers.concatenate([x, skip.pop()], axis=-1)
        x = keras.layers.Conv2D(filters=8*(2**i), kernel_size=(3, 3), padding='same', activation='relu')(x)
        x = keras.layers.UpSampling2D()(x)
    x = keras.layers.Conv2D(filters=1, kernel_size=(3, 3), padding='same')(x)
    keras_model = keras.models.Model(data, x)
    verify_keras_frontend(keras_model)


def test_forward_vgg16_layer():
    print("test_forward_vgg16_layer")
    keras_model = keras.applications.vgg16.VGG16(include_top=True, weights=None,
        input_shape=(224,224,3), classes=1000)
    data = keras.layers.Input((224,224,3))
    keras_model = keras.models.Model(data, keras_model(data))
    verify_keras_frontend(keras_model)


if __name__ == '__main__':
    test_forward_softrelu()
    test_forward_leaky_relu()
    test_forward_multiply()
    test_forward_dense()
    test_forward_conv_small()
    test_forward_conv()
    test_forward_transpose_conv()
    test_forward_depthwise_conv()
    test_forward_separable_conv()
    test_forward_upsample()
    test_forward_pooling()

    test_forward_seblock()
    test_forward_vgg16()
    test_forward_xception()
    test_forward_resnet50()
    test_forward_unet()
    test_forward_vgg16_layer()

    test_forward_shape_inference()


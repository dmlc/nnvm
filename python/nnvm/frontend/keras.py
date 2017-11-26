# pylint: disable=invalid-name, import-self
"""Keras frontend."""
from __future__ import absolute_import as _abs
import numpy as np
import six
import tvm
import keras as _keras
from .. import symbol as _sym
from .common import SymbolTable

__all__ = ['from_keras']


def _check_data_format(keras_layer):
    if hasattr(keras_layer, ('data_format')):
        if keras_layer.data_format != 'channels_last':
            raise ValueError("Keras frontend currently supports data_format = 'channels_last' only.")


def convert_activation(insym, keras_layer, symtab):
    if isinstance(keras_layer, _keras.layers.advanced_activations.LeakyReLU):
        return _sym.leaky_relu(insym, alpha=keras_layer.alpha)
    elif isinstance(keras_layer, _keras.layers.advanced_activations.ELU):
        return _sym.__mul_scalar__(_sym.__add_scalar__(_sym.exp(insym), scalar=-1),
            scalar=keras_layer.alpha)
    elif isinstance(keras_layer, _keras.layers.advanced_activations.PReLU):
        raise NotImplementedError('PReLU not implemented')
    elif isinstance(keras_layer, _keras.layers.advanced_activations.ThresholdedReLU):
        raise NotImplementedError('ThresholdedReLU not implemented')
    else:
        import six
        if six.PY2:
            act_name = keras_layer.activation.func_name
        else:
            act_name = keras_layer.activation.__name__

        if act_name == 'softmax':
            return _sym.softmax(insym)
        elif act_name == 'sigmoid':
            return _sym.sigmoid(insym)
        elif act_name == 'tanh':
            return _sym.tanh(insym)
        elif act_name == 'relu':
            return _sym.relu(insym)
        elif act_name == 'softplus':
            return _sym.log(_sym.__add_scalar__(_sym.exp(insym), scalar=1))
        elif act_name == 'elu':
            return _sym.__mul_scalar__(_sym.__add_scalar__(_sym.exp(insym),
                scalar=-1), scalar=1)
        elif act_name == 'linear':
            return _sym.__add_scalar__(_sym.__mul_scalar__(insym,
                scalar=keras_layer.alpha), scalar=keras_layer.beta)
        elif act_name == 'relu6':
            raise NotImplementedError('relu6 not implemented')
        elif act_name == 'softsign':
            raise NotImplementedError('softsign not implemented')
        elif act_name == 'hard_sigmoid':
            raise NotImplementedError('hard_sigmoid not implemented')
        else:
            raise TypeError("Unsupported activation type : {}".format(act_name))


def convert_dense(insym, keras_layer, symtab):
    weightList = keras_layer.get_weights()
    weight = symtab.new_const(weightList[0].transpose([1,0]))
    params = {'weight':weight, 'use_bias':False, 'units':weightList[0].shape[1]}
    if keras_layer.use_bias:
        params['use_bias'] = True
        bias = symtab.new_const(weightList[1])
        params['bias'] = bias
    dense = _sym.dense(data=insym, **params)
    # defuse activation
    act_name = keras_layer.activation.func_name if six.PY2 else keras_layer.activation.__name__
    if act_name != 'linear':
        act_layer = _keras.layers.core.Activation(act_name)
        dense = convert_activation(dense, act_layer, symtab)
    return dense


def convert_convolution(insym, keras_layer, symtab):
    # TODO: same padding fails when strides > 1
    _check_data_format(keras_layer)
    is_deconv = isinstance(keras_layer, _keras.layers.convolutional.Conv2DTranspose)
    weightList = keras_layer.get_weights()
    if is_deconv:
        height, width, n_filters, in_channels = weightList[0].shape
        weight = weightList[0].transpose([3,2,0,1])
    else:
        height, width, in_channels, n_filters = weightList[0].shape
        weight = weightList[0].transpose([3,2,0,1])
    dilation = [1,1]
    if (type(keras_layer.dilation_rate) is list) or (type(keras_layer.dilation_rate) is tuple):
        dilation = [keras_layer.dilation_rate[0], keras_layer.dilation_rate[1]]
    else:
        dilation = [keras_layer.dilation_rate, keras_layer.dilation_rate]
    groups = 1
    if isinstance(keras_layer, _keras.applications.mobilenet.DepthwiseConv2D):
        groups = in_channels
        depth_multiplier = keras_layer.depth_multiplier
        weight = np.reshape(weight, (in_channels, depth_multiplier, height, width))
    params = {'weight': symtab.new_const(weight),
              'channels': n_filters,
              'kernel_size': [height, width],
              'strides': list(keras_layer.strides),
              'dilation': dilation,
              'groups': groups}
    if keras_layer.use_bias:
        params['use_bias'] = True
        params['bias'] = symtab.new_const(weightList[1])
    if keras_layer.padding == 'valid':
        params['padding'] = (0,0)
    elif keras_layer.padding == 'same':
        kernel = params['kernel_size']
        pad_h = kernel[0] - 1
        pad_w = kernel[1] - 1
        pad_t = pad_h // 2
        pad_l = pad_w // 2
        pad_b = pad_h - pad_t
        pad_r = pad_w - pad_l
        assert pad_t == pad_r and pad_l == pad_b, "Asymmetry padding not supported"
        params['padding'] = [pad_t, pad_l]
    else:
        raise TypeError("Unsupported padding type : {}".format(keras_layer.padding))
    if is_deconv:
        out = _sym.conv2d_transpose(data=insym, **params)
    else:
        out = _sym.conv2d(data=insym, **params)
    # defuse activation
    act_name = keras_layer.activation.func_name if six.PY2 else keras_layer.activation.__name__
    if act_name != 'linear':
        act_layer = _keras.layers.core.Activation(act_name)
        out = convert_activation(out, act_layer, symtab)
    return out


def convert_separable_convolution(insym, keras_layer, symtab):
    _check_data_format(keras_layer)
    weightList = keras_layer.get_weights()
    height, width, in_channels, depth_mult = weightList[0].shape
    weight0 = weightList[0].transpose([2,3,0,1])
    params0 = {'weight': symtab.new_const(weight0),
               'channels': in_channels * depth_mult,
               'groups': in_channels,
               'kernel_size': [height, width],
               'strides': list(keras_layer.strides),
               'dilation': [1,1]}
    if keras_layer.padding == 'valid':
        params0['padding'] = (0,0)
    elif keras_layer.padding == 'same':
        kernel = params['kernel_size']
        pad_h = kernel[0] - 1
        pad_w = kernel[1] - 1
        pad_t = pad_h // 2
        pad_l = pad_w // 2
        pad_b = pad_h - pad_t
        pad_r = pad_w - pad_l
        assert pad_t == pad_r and pad_l == pad_b, "Asymmetry padding not supported"
        params0['padding'] = [pad_t, pad_l]
    else:
        raise TypeError("Unsupported padding type : {}".format(keras_layer.padding))
    depthwise_conv = _sym.conv2d(data=insym, **params0)

    weight1 = weightList[1].transpose([3,2,0,1])
    params1 = {'weight': symtab.new_const(weight1),
               'channels': weight1.shape[0],
               'groups': 1,
               'kernel_size': [1,1],
               'strides': [1,1],
               'dilation': [1,1]}
    if keras_layer.use_bias:
        params1['use_bias'] = True
        params1['bias'] = symtab.new_const(weightList[2])
    pointwise_conv = _sym.conv2d(data=depthwise_conv, **params1)
    # defuse activation
    act_name = keras_layer.activation.func_name if six.PY2 else keras_layer.activation.__name__
    if act_name != 'linear':
        act_layer = _keras.layers.core.Activation(act_name)
        pointwise_conv = convert_activation(pointwise_conv, act_layer, symtab)
    return pointwise_conv


def convert_pooling(insym, keras_layer, symtab):
    _check_data_format(keras_layer)
    if isinstance(keras_layer, _keras.layers.pooling.GlobalMaxPooling2D):
        return _sym.global_max_pool2d(insym)
    elif isinstance(keras_layer, _keras.layers.pooling.GlobalAveragePooling2D):
        return _sym.global_avg_pool2d(insym)
    else:
        params = {'pool_size':keras_layer.pool_size,
                  'strides':keras_layer.strides,
                  'padding':(1,1)}
        if keras_layer.padding == 'valid':
            params['padding'] = (0,0)
        elif keras_layer.padding == 'same':
            kernel = params['kernel_size']
            pad_h = kernel[0] - 1
            pad_w = kernel[1] - 1
            pad_t = pad_h // 2
            pad_l = pad_w // 2
            pad_b = pad_h - pad_t
            pad_r = pad_w - pad_l
            assert pad_t == pad_r and pad_l == pad_b, "Asymmetry padding not supported"
            params['padding'] = [pad_t, pad_l]
        else:
            raise TypeError("Unsupported padding type : {}".format(keras_layer.padding))
        if isinstance(keras_layer, _keras.layers.convolutional.MaxPooling2D):
            return _sym.max_pool2d(insym, **params)
        elif isinstance(keras_layer, _keras.layers.convolutional.AveragePooling2D):
            return _sym.avg_pool2d(insym, **params)
        else:
            raise TypeError("Unsupported pooling type : {}".format(keras_layer))


def convert_batchnorm(insym, keras_layer, symtab):
    params = {'gamma': symtab.new_const(keras_layer.get_weights()[0]),
              'beta': symtab.new_const(keras_layer.get_weights()[1]),
              'moving_mean': symtab.new_const(keras_layer.get_weights()[2]),
              'moving_var': symtab.new_const(keras_layer.get_weights()[3]),
              'epsilon': keras_layer.epsilon}
    return _sym.batch_norm(data=insym, **params)


def convert_flatten(insym, keras_layer, symtab):
    _check_data_format(keras_layer)
    insym = _sym.transpose(insym, axes=(0,2,3,1))
    return _sym.flatten(insym)


def convert_concat(insym, keras_layer, symtab):
    _check_data_format(keras_layer)
    assert(keras_layer.axis == -1 or keras_layer.axis == 3)
    if not isinstance(insyms, list):
        insyms = [insyms]
    return _sym.concatenate(*insyms, axis=1)


def convert_reshape(insym, keras_layer, symtab):
    return _sym.reshape(insym, keras_layer.shape)


def default_skip(insym, keras_layer, symtab):
    """Layers that can be skipped because they are train time only."""
    return


_convert_map  = {
    _keras.layers.core.Dense                           : convert_dense,
    _keras.layers.core.Activation                      : convert_activation,
    _keras.layers.advanced_activations.LeakyReLU       : convert_activation,
    _keras.layers.advanced_activations.PReLU           : convert_activation,
    _keras.layers.advanced_activations.ELU             : convert_activation,
    _keras.layers.advanced_activations.ThresholdedReLU : convert_activation,

    _keras.layers.pooling.AveragePooling2D             : convert_pooling,
    _keras.layers.pooling.MaxPooling2D                 : convert_pooling,
    _keras.layers.pooling.GlobalAveragePooling2D       : convert_pooling,
    _keras.layers.pooling.GlobalMaxPooling2D           : convert_pooling,
    _keras.layers.convolutional.Conv2D                 : convert_convolution,
    _keras.layers.convolutional.Conv2DTranspose        : convert_convolution,
    _keras.applications.mobilenet.DepthwiseConv2D      : convert_convolution,
    _keras.layers.convolutional.SeparableConv2D        : convert_separable_convolution,

    _keras.layers.Concatenate                          : convert_concat,
    _keras.layers.core.Flatten                         : convert_flatten,
    _keras.layers.core.Reshape                         : convert_reshape,
    _keras.layers.normalization.BatchNormalization     : convert_batchnorm,

    # _keras.layers.convolutional.ZeroPadding2D        : convert_padding,
    # _keras.layers.convolutional.Cropping2D           : convert_cropping,
    # _keras.layers.convolutional.UpSampling2D         : convert_upsample,
    # _keras.layers.pooling.AveragePooling1D           : convert_pooling,
    # _keras.layers.pooling.MaxPooling1D               : convert_pooling,
    # _keras.layers.pooling.GlobalAveragePooling1D     : convert_pooling,
    # _keras.layers.pooling.GlobalMaxPooling1D         : convert_pooling,
    # _keras.layers.convolutional.ZeroPadding1D        : convert_padding,
    # _keras.layers.convolutional.Cropping1D           : convert_cropping,
    # _keras.layers.convolutional.UpSampling1D         : convert_upsample,
    # _keras.layers.convolutional.Conv1D               : convert_convolution1d,

    # _keras.layers.recurrent.GRU                      : convert_gru,
    # _keras.layers.recurrent.LSTM                     : convert_lstm,
    # _keras.layers.recurrent.SimpleRNN                : convert_simple_rnn,
    # _keras.layers.wrappers.Bidirectional             : convert_bidirectional,
    # _keras.layers.wrappers.TimeDistributed           : default_skip,

    # _keras.layers.Add                                : convert_merge,
    # _keras.layers.Multiply                           : convert_merge,
    # _keras.layers.Average                            : convert_merge,
    # _keras.layers.Maximum                            : convert_merge,
    # _keras.layers.Dot                                : convert_merge,
    # _keras.layers.core.Permute                       : convert_permute,
    # _keras.layers.embeddings.Embedding               : convert_embedding,
    # _keras.layers.core.RepeatVector                  : convert_repeat_vector,

    _keras.engine.topology.InputLayer                  : default_skip,
    _keras.layers.core.Dropout                         : default_skip,
    _keras.layers.core.SpatialDropout2D                : default_skip,
    _keras.layers.core.SpatialDropout1D                : default_skip,
}


def _check_unsupported_layers(model):
    for layer in model.layers:
        if type(layer) not in _convert_map:
            raise ValueError(
                "Keras layer '%s' not supported. " % str(type(layer)))


def keras_op_to_nnvm(insym, keras_layer, outname, symtab):
    """Convert keras layer to nnvm symbol, and update symtab.

    Parameters
    ----------
    insym : nnvm.symbol.Symbol
        The input nnvm symbol

    keras_layer : keras.layers
        The keras layer to be converted

    outname : str
        Name of the output nnvm symbol

    symtab : nnvm.frontend.common.SymbolTable
        The global symbol table to be updated
    """
    if type(keras_layer) not in _convert_map:
        raise NotImplementedError("%s is not supported" % (type(keras_layer).__name__))
    ret = _convert_map[type(keras_layer)](insym, keras_layer, symtab)
    symtab.set_var(outname, ret)


def from_keras(model):
    """Convert keras model to NNVM format.

    Parameters
    ----------
    model : keras.engine.training.Model
        The keras model to be converted

    Returns
    -------
    sym : nnvm.Symbol
        Compatible nnvm symbol

    params : dict of str to tvm.NDArray
        The parameter dict to be used by nnvm
    """
    if _keras.backend.image_data_format() != 'channels_last':
        raise ValueError("Keras frontend currently supports data_format = channels_last only.")
    _check_unsupported_layers(model)

    symtab = SymbolTable()
    for keras_layer in model.layers:
        if isinstance(keras_layer, _keras.engine.topology.InputLayer):
            keras_layer.name = 'data'
            symtab.get_var(keras_layer.name, must_contain=False)
        else:
            predecessors = []
            for node in keras_layer.inbound_nodes:
                for pred in node.inbound_layers:
                    predecessors.append(pred.name)
            insym = symtab.get_var(predecessors[0], must_contain=True)
            keras_op_to_nnvm(insym, keras_layer, keras_layer.name, symtab)

    returns = [symtab.get_var(i.name, must_contain=False) for i in model.output_layers]
    tvmparams = {k:tvm.nd.array(np.array(v, dtype=np.float32)) for k, v in symtab.params.items()}
    return returns[0], tvmparams

# pylint: disable=invalid-name, import-self
"""Keras frontend."""
from __future__ import absolute_import as _abs
import numpy as np
import tvm
import sys
from .. import symbol as _sym
from .common import SymbolTable

__all__ = ['from_keras']


def _check_data_format(keras_layer):
    if hasattr(keras_layer, ('data_format')):
        if keras_layer.data_format != 'channels_last':
            raise ValueError("Keras frontend currently supports data_format = channels_last only.")


def convert_activation(insym, keras_layer, symtab):
    if isinstance(keras_layer, (str, unicode)):
        act_type = keras_layer
    else:
        if sys.version_info.major < 3:
            act_type = keras_layer.activation.func_name
        else:
            act_type = keras_layer.activation.__name__
    if act_type == 'linear':
        if isinstance(keras_layer, (str, unicode)):
            return insym
        else:
            return _sym.__add_scalar__(_sym.__mul_scalar__(insym,
                scalar=keras_layer.alpha), scalar=keras_layer.beta)
    elif act_type == 'softmax':
        return _sym.softmax(insym)
    elif act_type == 'sigmoid':
        return _sym.sigmoid(insym)
    elif act_type == 'tanh':
        return _sym.tanh(insym)
    elif act_type == 'relu':
        return _sym.relu(insym)
    elif act_type == 'softplus':
        return _sym.log(_sym.__add_scalar__(_sym.exp(insym), scalar=1))
    elif act_type == 'elu':
        return _sym.__mul_scalar__(_sym.__add_scalar__(_sym.exp(insym), scalar=-1), scalar=1)
    elif act_type == 'relu6':
        raise NotImplementedError('relu6 not implemented')
    elif act_type == 'softsign':
        raise NotImplementedError('softsign not implemented')
    elif act_type == 'hard_sigmoid':
        raise NotImplementedError('hard_sigmoid not implemented')
    else:
        raise TypeError("Unsupported activation type : {}".format(act_type))


def convert_advanced_activation(insym, keras_layer, symtab):
    act_type = type(keras_layer).__name__
    if act_type == 'LeakyReLU':
        return _sym.leaky_relu(insym, alpha=keras_layer.alpha)
    elif act_type == 'ELU':
        return _sym.__mul_scalar__(_sym.__add_scalar__(_sym.exp(insym), scalar=-1),
            scalar=keras_layer.alpha)
    elif act_type == 'PReLU':
        raise NotImplementedError('PReLU not implemented')
    elif act_type == 'ThresholdedReLU':
        raise NotImplementedError('ThresholdedReLU not implemented')
    else:
        raise TypeError("Unsupported advanced activation type : {}".format(act_type))


def convert_merge(insym, keras_layer, symtab):
    merge_type = type(keras_layer).__name__
    if merge_type == 'Add':
        return _sym.elemwise_add(insym[0], insym[1])
    elif merge_type == 'Subtract':
        return _sym.elemwise_sub(insym[0], insym[1])
    elif merge_type == 'Multiply':
        return _sym.elemwise_mul(insym[0], insym[1])
    else:
        raise TypeError("Unsupported merge type : {}".format(merge_type))


def convert_dense(insym, keras_layer, symtab):
    # insert flatten before dense when necessary
    if 'pool' in insym.list_output_names()[0]:
        insym = convert_flatten(insym, keras_layer, symtab)
    # print dir(insym)
    # print insym.list_attr()
    # print insym.get_internals()
    # print insym.list_input_names()
    weightList = keras_layer.get_weights()
    weight = symtab.new_const(weightList[0].transpose([1,0]))
    params = {'weight':weight, 'use_bias':False, 'units':weightList[0].shape[1]}
    if keras_layer.use_bias:
        params['use_bias'] = True
        params['bias'] = symtab.new_const(weightList[1])
    dense = _sym.dense(data=insym, **params)
    # defuse activation
    if sys.version_info.major < 3:
        act_type = keras_layer.activation.func_name
    else:
        act_type = keras_layer.activation.__name__
    if act_type != 'linear':
        dense = convert_activation(dense, act_type, symtab)
    return dense


def convert_convolution(insym, keras_layer, symtab):
    # TODO: same padding fails when strides > 1
    _check_data_format(keras_layer)
    is_deconv = type(keras_layer).__name__ == 'Conv2DTranspose'
    weightList = keras_layer.get_weights()
    if is_deconv:
        kernel_h, kernel_w, n_filters, in_channels = weightList[0].shape
        weight = weightList[0].transpose([3,2,0,1])
    else:
        kernel_h, kernel_w, in_channels, n_filters = weightList[0].shape
        weight = weightList[0].transpose([3,2,0,1])
    dilation = [1,1]
    if (type(keras_layer.dilation_rate) is list) or (type(keras_layer.dilation_rate) is tuple):
        dilation = [keras_layer.dilation_rate[0], keras_layer.dilation_rate[1]]
    else:
        dilation = [keras_layer.dilation_rate, keras_layer.dilation_rate]
    groups = 1
    if type(keras_layer).__name__ == 'DepthwiseConv2D':
        groups = in_channels
        depth_multiplier = keras_layer.depth_multiplier
        weight = np.reshape(weight, (in_channels, depth_multiplier, kernel_h, kernel_w))
    params = {'weight': symtab.new_const(weight),
              'channels': n_filters,
              'kernel_size': [kernel_h, kernel_w],
              'strides': list(keras_layer.strides),
              'dilation': dilation,
              'groups': groups}
    if keras_layer.use_bias:
        params['use_bias'] = True
        params['bias'] = symtab.new_const(weightList[1])
    if keras_layer.padding == 'valid':
        params['padding'] = (0,0)
    elif keras_layer.padding == 'same':
        pad_h = kernel_h - 1
        pad_w = kernel_w - 1
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
    if sys.version_info.major < 3:
        act_type = keras_layer.activation.func_name
    else:
        act_type = keras_layer.activation.__name__
    if act_type != 'linear':
        out = convert_activation(out, act_type, symtab)
    return out


def convert_separable_convolution(insym, keras_layer, symtab):
    _check_data_format(keras_layer)
    weightList = keras_layer.get_weights()
    kernel_h, kernel_w, in_channels, depth_mult = weightList[0].shape
    weight0 = weightList[0].transpose([2,3,0,1])
    params0 = {'weight': symtab.new_const(weight0),
               'channels': in_channels * depth_mult,
               'groups': in_channels,
               'kernel_size': [kernel_h, kernel_w],
               'strides': list(keras_layer.strides),
               'dilation': [1,1]}
    if keras_layer.padding == 'valid':
        params0['padding'] = (0,0)
    elif keras_layer.padding == 'same':
        pad_h = kernel_h - 1
        pad_w = kernel_w - 1
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
    if sys.version_info.major < 3:
        act_type = keras_layer.activation.func_name
    else:
        act_type = keras_layer.activation.__name__
    if act_type != 'linear':
        pointwise_conv = convert_activation(act_type, act_type, symtab)
    return pointwise_conv


def convert_pooling(insym, keras_layer, symtab):
    _check_data_format(keras_layer)
    pool_type = type(keras_layer).__name__
    if pool_type == 'GlobalMaxPooling2D':
        return _sym.global_max_pool2d(insym)
    elif pool_type == 'GlobalAveragePooling2D':
        return _sym.global_avg_pool2d(insym)
    else:
        params = {'pool_size':keras_layer.pool_size,
                  'strides':keras_layer.strides,
                  'padding':(1,1)}
        if keras_layer.padding == 'valid':
            params['padding'] = (0,0)
        elif keras_layer.padding == 'same':
            pool_size = params['pool_size']
            pad_h = pool_size[0] - 1
            pad_w = pool_size[1] - 1
            pad_t = pad_h // 2
            pad_l = pad_w // 2
            pad_b = pad_h - pad_t
            pad_r = pad_w - pad_l
            assert pad_t == pad_r and pad_l == pad_b, "Asymmetry padding not supported"
            params['padding'] = [pad_t, pad_l]
        else:
            raise TypeError("Unsupported padding type : {}".format(keras_layer.padding))
        if pool_type == 'MaxPooling2D':
            return _sym.max_pool2d(insym, **params)
        elif pool_type == 'AveragePooling2D':
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
    'Dense'                    : convert_dense,
    'Activation'               : convert_activation,
    'LeakyReLU'                : convert_advanced_activation,
    'PReLU'                    : convert_advanced_activation,
    'ELU'                      : convert_advanced_activation,
    'ThresholdedReLU'          : convert_advanced_activation,

    'AveragePooling2D'         : convert_pooling,
    'MaxPooling2D'             : convert_pooling,
    'GlobalAveragePooling2D'   : convert_pooling,
    'GlobalMaxPooling2D'       : convert_pooling,
    'Conv2D'                   : convert_convolution,
    'Conv2DTranspose'          : convert_convolution,
    'DepthwiseConv2D'          : convert_convolution,
    'SeparableConv2D'          : convert_separable_convolution,

    'Flatten'                  : convert_flatten,
    'Reshape'                  : convert_reshape,
    'Concatenate'              : convert_concat,
    'BatchNormalization'       : convert_batchnorm,

    'Add'                      : convert_merge,
    'Subtract'                 : convert_merge,
    'Multiply'                 : convert_merge,

    # 'ZeroPadding1D'          : convert_padding,
    # 'ZeroPadding2D'          : convert_padding,
    # 'AveragePooling1D'       : convert_pooling,
    # 'MaxPooling1D'           : convert_pooling,
    # 'GlobalAveragePooling1D' : convert_pooling,
    # 'GlobalMaxPooling1D'     : convert_pooling,
    # 'Cropping1D'             : convert_cropping,
    # 'Cropping2D'             : convert_cropping,
    # 'UpSampling1D'           : convert_upsample,
    # 'UpSampling2D'           : convert_upsample,
    # 'Conv1D'                 : convert_convolution1d,

    # 'GRU'                    : convert_gru,
    # 'LSTM'                   : convert_lstm,
    # 'SimpleRNN'              : convert_simple_rnn,
    # 'Bidirectional'          : convert_bidirectional,
    # 'TimeDistributed'        : default_skip,

    # 'Average'                : convert_merge,
    # 'Maximum'                : convert_merge,
    # 'Dot'                    : convert_merge,
    # 'Permute'                : convert_permute,
    # 'Embedding'              : convert_embedding,
    # 'RepeatVector'           : convert_repeat_vector,

    'InputLayer'               : default_skip,
    'Dropout'                  : default_skip,
    'SpatialDropout2D'         : default_skip,
    'SpatialDropout1D'         : default_skip,
}


def _check_unsupported_layers(model):
    for layer in model.layers:
        if type(layer).__name__ not in _convert_map:
            raise ValueError("Keras layer {} not supported.".format(type(layer).__name__))


def keras_op_to_nnvm(insym, keras_layer, outname, symtab):
    """Convert keras layer to nnvm symbol, and update symtab.

    Parameters
    ----------
    insym : nnvm.symbol.Symbol or a list of it
        The input nnvm symbol(s)

    keras_layer : keras.layers
        The keras layer to be converted

    outname : str
        Name of the output nnvm symbol

    symtab : nnvm.frontend.common.SymbolTable
        The global symbol table to be updated
    """
    if type(keras_layer).__name__ not in _convert_map:
        raise NotImplementedError("{} is not supported".format((type(keras_layer).__name__)))
    ret = _convert_map[type(keras_layer).__name__](insym, keras_layer, symtab)
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
    try:
        import keras
    except ImportError:
        raise ImportError('Keras must be installed')
    assert isinstance(model, keras.engine.training.Model)
    if keras.backend.image_data_format() != 'channels_last':
        raise ValueError("Keras frontend currently supports data_format = channels_last only.")
    _check_unsupported_layers(model)

    # for i in model.layers:
    #     print i.name

    symtab = SymbolTable()
    for keras_layer in model.layers:
        if isinstance(keras_layer, keras.engine.topology.InputLayer):
            keras_layer.name = 'data'
            symtab.get_var(keras_layer.name, must_contain=False)
        else:
            predecessors = []
            for node in keras_layer.inbound_nodes:
                for pred in node.inbound_layers:
                    predecessors.append(pred.name)
            if len(predecessors) == 1:
                insym = symtab.get_var(predecessors[0], must_contain=True)
            else:
                insym = [symtab.get_var(pred, must_contain=True) for pred in predecessors]
            print predecessors
            keras_op_to_nnvm(insym, keras_layer, keras_layer.name, symtab)

    returns = [symtab.get_var(i.name, must_contain=False) for i in model.output_layers]
    tvmparams = {k:tvm.nd.array(np.array(v, dtype=np.float32)) for k, v in symtab.params.items()}
    return returns[0], tvmparams

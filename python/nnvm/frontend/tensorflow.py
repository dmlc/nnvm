# pylint: disable=import-self, invalid-name, unused-argument
"""TF: Tensorflow frontend."""
from __future__ import absolute_import as _abs

# Numpy support
import numpy as np

import tvm
from .. import symbol as _sym
from .. import graph as _graph
from .. compiler import graph_util
from .common import get_nnvm_op, AttrConverter as AttrConvert

__all__ = ['from_tensorflow']

class AttrCvt(object):
    """A Wrapper to handle some common jobs:
    """
    def __init__(self, op_name, transforms=None,
                 excludes=None, disables=None, ignores=None,
                 extras=None, custom_check=None):
        self._op_name = op_name
        self._transforms = transforms if transforms else {}
        self._excludes = excludes if excludes else []
        self._disables = disables if disables else []
        self._ignores = ignores if ignores else []
        self._extras = extras if extras else {}
        self._custom_check = custom_check

    def __call__(self, inputs, attrs, *args):
        self._ignores.append('_output_shapes')
        self._ignores.append('_input_shapes')
        self._ignores.append('T')
        self._ignores.append('use_cudnn_on_gpu')
        return AttrConvert(self._op_name, self._transforms, self._excludes,
                           self._disables, self._ignores, self._extras,
                           self._custom_check)(inputs, attrs, *args)

def _get_pad_pair(input1d, kernel1d, stride1d):
    if input1d % stride1d == 0:
        pad = max(kernel1d - stride1d, 0)
    else:
        pad = max(kernel1d - (input1d % stride1d), 0)

    pad_before = pad // 2
    pad_after = pad - pad_before

    return [pad_before, pad_after]

def _math_name_picker(surfix):
    def _impl(attr):
        return 'broadcast_' + surfix
    return _impl

def _dimension_picker(prefix, surfix=''):
    def _impl(attr):
        kernel = attr['kernel_shape']
        if len(kernel) == 2:
            return prefix + '2d' + surfix
        else:
            raise NotImplementedError("Only 2d kernel supported.")
    return _impl

def _dimension_constraint():
    def _dim_check(attrs):
        if len(attrs['kernel_shape']) == 2:
            return True
        return False
    return _dim_check, "Only 2d kernel supported."

def _infer_channels(inputs, params, transpose=False):
    """A hack for getting 'channles' or 'units' since tensorflow don't provide
    these attributes. We check the shape of weights provided to get the number.
    """
    g = _graph.create(inputs)
    shape_dict = {k: v.shape for k, v in params.items()}
    _, out_shapes = graph_util.infer_shape(g, **shape_dict)
    channels = out_shapes[0][0] if not transpose else out_shapes[0][1]
    return channels

def _elemwise(name):
    def _impl(inputs, attr, *args):
        assert len(inputs) == 2, "Math op take 2 inputs, {} given".format(len(inputs))
        op_name = _math_name_picker(name)(attr)
        axis = int(attr.get('axis', 0))
        conv_ops = ["conv2d", "conv2d_transpose"]
        if op_name == 'broadcast_add' and inputs[0].attr('op_name') in conv_ops:
            # TODO: remove hard coded infershape
            inputs[1] = _sym.expand_dims(inputs[1], axis=axis, num_newaxis=2)
        return get_nnvm_op(op_name)(*inputs)
    return _impl

def _pooling(name):
    def _impl(inputs, attr, params):

        attr['data_format'] = attr['data_format'].decode("utf-8")

        if attr['data_format'] == 'NHWC':
            attr['kernel_shape'] = (attr['ksize'][1], attr['ksize'][2])
        elif attr['data_format'] == 'NCHW':
            attr['kernel_shape'] = (attr['ksize'][2], attr['ksize'][3])
        else:
            raise TypeError("Unsupported data_format type : {}".format(attr['data_format']))

        # Fix strides
        attr['strides'] = (attr['strides'][1], attr['strides'][2])

        # Fix padding
        input_shapes = attr['_input_shapes'][inputs[0]]
        attr['padding'] = attr['padding'].decode("utf-8")

        if attr['padding'] == 'VALID':
            attr['padding'] = [0, 0]
        elif attr['padding'] == 'SAME':
            stride_h, stride_w = attr['strides']
            kernel_h, kernel_w = attr['kernel_shape']
            if attr['data_format'] == 'NHWC':
                in_h = input_shapes[0][1]
                in_w = input_shapes[0][2]
            else:
                in_h = input_shapes[0][2]
                in_w = input_shapes[0][3]

            pad_v = _get_pad_pair(in_h, kernel_h, stride_h)
            pad_h = _get_pad_pair(in_w, kernel_w, stride_w)

            if attr['data_format'] == 'NHWC':
                inputs[0] = _sym.pad(data=inputs[0],
                                     pad_width=((0, 0),
                                                (pad_v[0], pad_v[1]),
                                                (pad_h[0], pad_h[1]),
                                                (0, 0)))
            else:
                inputs[0] = _sym.pad(data=inputs[0],
                                     pad_width=((0, 0),
                                                (0, 0),
                                                (pad_v[0], pad_v[1]),
                                                (pad_h[0], pad_h[1])))
            attr['padding'] = [0, 0]
        else:
            raise TypeError("Unsupported padding type : {}".format(attr['padding']))

        return AttrCvt(
            op_name=_dimension_picker(name),
            transforms={
                'kernel_shape':'pool_size',
                'data_format':'layout'},
            ignores=['ksize'],
            extras={'ceil_mode': False},
            custom_check=_dimension_constraint())(inputs, attr)
    return _impl

def _conv():
    def _impl(inputs, attr, params):
        attr['data_format'] = attr['data_format'].decode("utf-8")

        # Extract kernel shape from params
        conv_param_weights = params[inputs[1].list_output_names()[0]]

        if attr['data_format'] == 'NHWC':
            attr['kernel_shape'] = (conv_param_weights.shape[0], conv_param_weights.shape[1])
            attr['channels'] = conv_param_weights.shape[3]
            if 'dilations' in attr:
                attr['dilations'] = (attr['dilations'][0], attr['dilations'][1])
        elif attr['data_format'] == 'NCHW':
            attr['kernel_shape'] = (conv_param_weights.shape[2], conv_param_weights.shape[3])
            attr['channels'] = conv_param_weights.shape[1]
            if 'dilations' in attr:
                attr['dilations'] = (attr['dilations'][2], attr['dilations'][3])
        else:
            raise TypeError("Unsupported data format type : {}".format(attr['data_format']))

        # Fix strides
        attr['strides'] = (attr['strides'][1], attr['strides'][2])

        # Fix padding
        input_shapes = attr['_input_shapes'][inputs[0]]
        attr['padding'] = attr['padding'].decode("utf-8")

        if attr['padding'] == 'VALID':
            attr['padding'] = [0, 0]
        elif attr['padding'] == 'SAME':
            stride_h, stride_w = attr['strides']
            kernel_h, kernel_w = attr['kernel_shape']
            if attr['data_format'] == 'NHWC':
                in_h = input_shapes[0][1]
                in_w = input_shapes[0][2]
            else:
                in_h = input_shapes[0][2]
                in_w = input_shapes[0][3]

            pad_v = _get_pad_pair(in_h, kernel_h, stride_h)
            pad_h = _get_pad_pair(in_w, kernel_w, stride_w)

            if attr['data_format'] == 'NHWC':
                inputs[0] = _sym.pad(data=inputs[0],
                                     pad_width=((0, 0),
                                                (pad_v[0], pad_v[1]),
                                                (pad_h[0], pad_h[1]),
                                                (0, 0)))
            else:
                inputs[0] = _sym.pad(data=inputs[0],
                                     pad_width=((0, 0),
                                                (0, 0),
                                                (pad_v[0], pad_v[1]),
                                                (pad_h[0], pad_h[1])))

            attr['padding'] = [0, 0]

        else:
            raise TypeError("Unsupported padding type : {}".format(attr['padding']))

        if 'kernel_layout' not in attr:
            attr['kernel_layout'] = 'HWIO' if attr['data_format'] == 'NHWC' else 'OIHW'

        return AttrCvt(
            op_name=_dimension_picker('conv'),
            transforms={
                'kernel_shape': 'kernel_size',
                'data_format': 'layout',
                'dilations': ('dilation', (0, 0)),
                'group': ('groups', 1)},
            extras={'use_bias': len(inputs) == 3},
            custom_check=_dimension_constraint())(inputs, attr)
    return _impl

def _decode_image():
    def _impl(inputs, attr, params):
        # Image decode wrapper: Expecting user to feed decoded input to next layer drop this layer.
        return inputs[0]
    return _impl

def _cast():
    def _impl(inputs, attr, params):
        # Convert from tensorflow Dtype to str
        attr['DstT'] = attr['DstT'].name
        return AttrCvt(op_name='cast', transforms={'DstT': 'dtype'}, ignores=['SrcT'])(inputs, attr)
    return _impl

def _expand_dims():
    def _impl(inputs, attr, params):
        dim_input = inputs.pop(1)
        axis = params[dim_input.list_output_names()[0]]
        params.pop(dim_input.list_output_names()[0])
        return AttrCvt(op_name="expand_dims", ignores=['Tdim'],
                       extras={'axis': axis.asnumpy()[0]})(inputs, attr)
    return _impl

def _resize_bilinear():
    def _impl(inputs, attr, params):
        # Making a copy node assuming the input image shape is 299x299
        # Change this when we have corresponding resize bilinear operation.
        pop_node = inputs.pop(1)
        params.pop(pop_node.list_output_names()[0])
        return AttrCvt(op_name="copy", ignores=['align_corners'])(inputs, attr)
    return _impl

def _check_numerics():
    def _impl(inputs, attr, params):
        # Making a copy node assuming no need to verify
        return AttrCvt(op_name="copy", ignores=['message'])(inputs, attr)
    return _impl


def _matmul():
    def _impl(inputs, attr, params):
        channels = _infer_channels(inputs[1], params, not attr['transpose_b'])
        if attr['transpose_a']:
            inputs[0] = _sym.transpose(inputs[0], axis(1, 0))
        if not attr['transpose_b']:
            inputs[1] = _sym.transpose(inputs[1], axes=(1, 0))
        return AttrCvt(op_name="dense",
                       extras={'use_bias': False, 'units': channels},
                       ignores=['transpose_a', 'transpose_b', 'T'])(inputs, attr)

    return _impl

def _identity():
    def _impl(inputs, attr, params):
        # Tensorflow takes CheckNumerics as
        # second argument which we could ignore for time being.
        if len(inputs) == 2:
            pop_node = inputs.pop(1)
            params.pop(pop_node.list_output_names()[0])
        return AttrCvt(op_name="copy", ignores=['T'])(inputs, attr)
    return _impl

def _concatV2():
    def _impl(inputs, attr, params):
        pop_node = inputs.pop(len(inputs)-1)
        axis = params[pop_node.list_output_names()[0]]
        params.pop(pop_node.list_output_names()[0])
        return AttrCvt(
            op_name="concatenate", ignores=['T', 'N', 'Tidx', '_output_shapes'],
            extras={'axis': axis.asnumpy()[0]})(inputs, attr)
    return _impl

def _concat():
    def _impl(inputs, attr, params):
        pop_node = inputs.pop(0)
        axis = params[pop_node.list_output_names()[0]]
        params.pop(pop_node.list_output_names()[0])
        return AttrCvt(
            op_name="concatenate", ignores=['N'],
            extras={'axis': axis.asnumpy()[0]})(inputs, attr)
    return _impl

def _reshape():
    def _impl(inputs, attr, params):
        pop_node = inputs.pop(1)
        shape_arg = params[pop_node.list_output_names()[0]]
        params.pop(pop_node.list_output_names()[0])
        return AttrCvt(
            op_name="reshape",
            extras={'shape':tuple(shape_arg.asnumpy())},
            ignores=['Tshape'])(inputs, attr)
    return _impl

def _bias_add():
    def _impl(inputs, attr, params):
        return _sym.broadcast_add(inputs[0], inputs[1])
    return _impl

def _batch_norm():
    def _impl(inputs, attr, params):
        # Rearrange inputs from
        # (data, moving_mean, moving_variance, beta, gamma)
        #     to
        # (data, gamma, beta, moving_mean, moving_var)
        new_inputs = [inputs[0], inputs[4], inputs[3], inputs[1], inputs[2]]

        return AttrCvt(
            op_name='batch_norm',
            transforms={'scale_after_normalization':'scale', 'variance_epsilon':'epsilon'},
            extras={'axis': 3}, # Fix axis
            disables=['momentum'])(new_inputs, attr)
    return _impl

def _fill():
    def _impl(inputs, attr, params):
        '''pop_node = inputs.pop(1)
        fill_arg = params[pop_node.list_output_names()[0]]
        params.pop(pop_node.list_output_names()[0])
        new_inputs = []
        return AttrCvt(
                op_name='full',
                extras={'shape':inputs[0], 'fill_value':float(fill_arg.asnumpy()[0]), 'dtype':'int32'},
                ignores=['index_type', 'T', '_output_shapes'])(new_inputs, attr)'''
        fill_arg = params.pop(inputs.pop(1).list_output_names()[0])
        #shape_arg = params.pop(inputs.pop(0).list_output_names()[0])
        new_inputs = []
        return AttrCvt(
                op_name='full',
                extras={'shape':inputs[0],
                        'fill_value':fill_arg.asnumpy()[0], 'dtype':attr['T'].name},
                ignores=['index_type', 'T', '_output_shapes'])(new_inputs, attr)
    return _impl

def _init_state(in_state_c_name, in_state_h_name, num_layers, batch_size, num_hidden):
    """Create the initial states for the first layer in the graph."""
    in_state_c = _sym.Variable(in_state_c_name, shape=(num_layers, batch_size, num_hidden))
    in_state_h = _sym.Variable(in_state_h_name, shape=(num_layers, batch_size, num_hidden))
    return in_state_c, in_state_h

def _get_cur_input_state(in_state_c, in_state_h,
            num_layers, layer, batch_size, num_hidden):
    """Select the states for the current layer"""
    in_state_c_tup = _sym.split(in_state_c, indices_or_sections=num_layers, axis=0)
    in_state_h_tup = _sym.split(in_state_h, indices_or_sections=num_layers, axis=0)
    cur_in_state_c = _sym.reshape(in_state_c_tup[layer], shape=(batch_size, num_hidden))
    cur_in_state_h = _sym.reshape(in_state_h_tup[layer], shape=(batch_size, num_hidden))
    return cur_in_state_c, cur_in_state_h

def _LSTMBlockCell(inputs, in_state_c, in_state_h, attr, params):
    """LSTM Block cell.

    ixh = [x, h_prev]
    [i, ci, f, o] = xh * w + b
    f = f + forget_bias

    if not use_peephole:
    wci = wcf = wco = 0

    i = sigmoid(cs_prev * wci + i)
    f = sigmoid(cs_prev * wcf + f)
    ci = tanh(ci)

    cs = ci .* i + cs_prev .* f
    cs = clip(cs, cell_clip)

    o = sigmoid(cs * wco + o)
    co = tanh(cs)
    h = co .* o

    Parameters
    ----------
    inputs : nnvm.Symbol
        Input data
    in_state_c: list of nnvm.Symbol
        Cell state input values for all the layers
    in_state_h: list of nnvm.Symbol
        Hidden state input values for all the layers
    attrs : dict
        Dict of operator attributes
    params : dict
        List of  pretrained weights and bias

    Returns
    -------
    sym : nnvm.Symbol
        Converted nnvm Symbol
    output: nnvm.Symbol
        Output state value.
    """
    in_data = inputs[0]
    in_weight = inputs[3]
    in_bias = inputs[7]
    forget_bias = attr.pop('forget_bias')
    input_shape =  attr['_input_shapes'][inputs[0]]
    weight_shape =  attr['_input_shapes'][inputs[3]]
    batch_size, input_size = input_shape[0][0], input_shape[0][1]
    num_hidden_layers = weight_shape[0][1]
    num_hidden = num_hidden_layers / 4

    in_data = _sym.reshape(in_data, shape=(batch_size, input_size))
    ixh = _sym.concatenate(*[in_data, in_state_h], axis=1)
    in_weight = _sym.transpose(in_weight)
    gates = _sym.dense(ixh, in_weight, in_bias, use_bias=True,
                       units=num_hidden_layers, name="dense")
    gate_list = _sym.split(gates, indices_or_sections=4, axis=1)
    in_gate = _sym.sigmoid(gate_list[0])
    in_transform = _sym.tanh(gate_list[1])
    forget_gate = _sym.sigmoid(gate_list[2])
    forget_gate = forget_gate + forget_bias
    out_gate = _sym.sigmoid(gate_list[3])
    next_c = _sym.broadcast_add(_sym.broadcast_mul(forget_gate, in_state_c),
                                _sym.broadcast_mul(in_gate, in_transform))
    next_h = out_gate * _sym.tanh(next_c)
    out_state = _sym.concatenate(*[next_c, next_h])
    out_state = _sym.reshape(out_state, shape=(2, batch_size, num_hidden))
    return next_h, out_state

def _LSTMBlockCellWrapper():
    def _impl(op_name, inputs, in_state_c_name, in_state_h_name,
            in_state_c, in_state_h, attr, params, graph, layer):
        """Wrapper on LSTMBlockCell to maintain the LSTM states. Calaculate
        the number of layers of same operator and create input states for
        all the layers. Output states values are set to the global list.

        Parameters
        ----------
        op_name : str
            Operator name, such as Conv2D, AvgPool
        inputs : list of nnvm.Symbol
            List of input symbols.
        in_state_c_name: str
            Input cell state value
        in_state_h_name: str
            Hidden output state value
        in_state_c: list of nnvm.Symbol
            Cell state input values for all the layers
        in_state_h: list of nnvm.Symbol
            Hidden state input values for all the layers
        attrs : dict
            Dict of operator attributes
        params : dict
            List of  pretrained weights and bias
        graph : Tensorflow graph object
            Graph is to find the number of upcoming same node to
            calculate the number of layers.
        layer : int
            Current operator layer (count) in the graph

        Returns
        -------
        sym : nnvm.Symbol
            Converted nnvm Symbol
        output: nnvm.Symbol
            Output state value.
        in_state_c: nnvm.Symbol
            Input cell state placeholder created.
        in_state_h: nnvm.Symbol
            Input hidden state placeholder created.
        """
        input_shape =  attr['_input_shapes'][inputs[0]]
        weight_shape =  attr['_input_shapes'][inputs[3]]
        batch_size = input_shape[0][0]
        num_hidden = weight_shape[0][1] / 4

        #find number of layers of this same operator node in the graph
        num_layers = 0
        for node in graph.node:
            if node.op == op_name:
                num_layers +=1
        if layer == 0:
            #Create initial states placeholder in case of first layer
            in_state_c, in_state_h = _init_state(in_state_c_name, in_state_h_name,
                                                 num_layers, batch_size, num_hidden)
        cur_in_state_c, cur_in_state_h = _get_cur_input_state(in_state_c, in_state_h,
                                                              num_layers, layer,
                                                              batch_size, num_hidden)
        output, out_state = _LSTMBlockCell(inputs, cur_in_state_c,
                                           cur_in_state_h, attr, params)
        return output, out_state, in_state_c, in_state_h
    return _impl

def _Gather():
    def _impl(inputs, attr, params):
        #print(params[inputs[0].list_output_names()[0]])
        axis = int(attr.get('axis', 0))
        #print("_Gather(), axis:", axis, ", new_inputs:", new_inputs, ", attr:", attr, ", params:", params)
        return AttrCvt(
                op_name="take",
                extras={'axis':axis},
                ignores=['Tindices', 'Tparams', 'validate_indices', 'axis', '_output_shapes', 'Taxis', '_class'])(inputs, attr)
    return _impl

def _pack():
    def _impl(inputs, attr, params):
        axis = int(attr.get('axis', 0))
        if 1 == len(inputs):
            return _sym.expand_dims(inputs[0], axis=axis, num_newaxis=1)
        new_inputs = [_sym.expand_dims(input_, axis=axis, num_newaxis=1) for input_ in inputs]
        return AttrCvt(
            op_name="concatenate",
            extras={'axis':axis},
            ignores=['T', 'N', '_output_shapes', 'axis'])(new_inputs, attr)
    return _impl
def _infer_out_shapes(inputs, params, transpose=False):
    """A hack for getting 'channles' or 'units' since onnx don't provide
    these attributes. We check the shape of weights provided to get the number.
    """
    g = _graph.create(inputs)
    shape_dict = {k: v.shape for k, v in params.items()}
    _, out_shapes = graph_util.infer_shape(g, **shape_dict)
    return out_shapes
def _stridedSlice_():
    def _impl(inputs, attr, params):
        new_inputs = [[]]
        #print("input", params[inputs[1].list_output_names()[0]].asnumpy().size)
        begin_input = [params[inputs[1].list_output_names()[0]].asnumpy()[i] for i in range(params[inputs[1].list_output_names()[0]].asnumpy().size)]
        end_input = [params[inputs[2].list_output_names()[0]].asnumpy()[i] for i in range(params[inputs[2].list_output_names()[0]].asnumpy().size)]
        stride_input = [params[inputs[3].list_output_names()[0]].asnumpy()[i] for i in range(params[inputs[3].list_output_names()[0]].asnumpy().size)]

        new_inputs[0] = inputs[0]
        inputs_0_shape = _infer_out_shapes(inputs[0], params)

        # change to empy later
        '''begin = [0 for i in range(len(inputs_0_shape[0]))]
        end = [0 for i in range(len(inputs_0_shape[0]))]
        stride = [0 for i in range(len(inputs_0_shape[0]))]'''
        begin = begin_input
        end = end_input
        stride = stride_input
        #print("input", begin, end, stride)
        
        ellipsis_mask_inp = int(attr.get('ellipsis_mask', 0))
        shrink_axis_mask_inp = int(attr.get('shrink_axis_mask', 0))
        end_mask_inp = int(attr.get('end_mask', 0))
        begin_mask_inp = int(attr.get('begin_mask', 0))
        #new_axis_mask = int(attr.get('new_axis_mask', 0))
        #new_axis_mask_inp = 0

        ellipsis_mask = 0
        shrink_axis_mask = 0
        end_mask = 0
        begin_mask = 0
        new_axis_mask = 0
        #print("ellipsis_mask", ellipsis_mask, "shrink_axis_mask", shrink_axis_mask, "end_mask", end_mask, "begin_mask", begin_mask, "new_axis_mask", new_axis_mask)

        #print("inputs_0_shape :", inputs_0_shape)
        '''return AttrCvt(
                op_name="strided_slice",
                extras={'begin':begin, 'end':end, 'stride':stride},
                #extras={'fill_value':3.0, 'dtype':'int32'},
                ignores=['ellipsis_mask', 'index_type', 'T', '_output_shapes', 'axis', 'N', 'shrink_axis_mask', 'Index', 'end_mask', 'new_axis_mask', 'begin_mask'])(new_inputs, attr)'''
        '''begin = inputs.pop(1)
        end = inputs.pop(2)
        stride = inputs.pop(3)
        return get_nnvm_op("strided_slice")(*inputs, begin = begin, end = end, stride = stride)'''
        kShrinkAxis = -1
        kNewAxis = -2
        input_dim = len(inputs_0_shape[0])
        stride_dim = len(stride)
        ellipsis_seen = False
        num_add_axis_after_ellipsis = 0
        final_shape_gather_indices = []
        for i in range(stride_dim):
            if (ellipsis_seen and ((1 << i) & new_axis_mask) != 0):
                num_add_axis_after_ellipsis = num_add_axis_after_ellipsis + 1

            if (((1 << i) & ellipsis_mask) != 0):
                ellipsis_seen = True


        #// If no ellipsis insert one at the end
        if (not ellipsis_seen):
            ellipsis_mask_inp |= (1 << stride_dim);
            stride_dim = stride_dim + 1;

        def transform_ellipsis():
            full_index = 0;
            shrink_axis_mask = 0
            begin_mask = 0
            end_mask = 0
            
            for i in range(stride_dim):
                if ((1 << i) & ellipsis_mask_inp):
                    #// Expand the ellipsis into the appropriate indices
                    #// NOTE: this only works because we guaranteed one ellipsis
                    next_index = min(input_dim - (stride_dim - i) + 1 + num_add_axis_after_ellipsis,
                                          input_dim)
                    #for (; full_index < next_index; full_index++):
                    for full_index in range(full_index, next_index):
                        #// new_axis' aren't real axis so you have to skip
                        begin[full_index] = end[full_index] = 0;
                        stride[full_index] = 1;
                        begin_mask = begin_mask | (1 << full_index);
                        end_mask = end_mask | (1 << full_index);
                        final_shape_gather_indices.append(full_index);

                elif ((1 << i) & new_axis_mask):
                    final_shape_gather_indices.append(kNewAxis);
                else:
                    if (full_index == len(begin)):
                        return

                    #// Gather slicing spec into appropriate index
                    if (len(begin_input) != 0):
                        begin[full_index] = begin_input[i]

                    if (len(end_input) != 0):
                        end[full_index] = end_input[i]

                    stride[full_index] = stride_input[i]

                    if (begin_mask_inp & (1 << i)):
                        begin_mask |= (1 << full_index);

                    if (end_mask_inp & (1 << i)):
                        end_mask |= (1 << full_index);

                    #// If shrink, record where to get the dimensionality from (i.e.
                    #// new_axis creates a fake 1 size dimension. Also remember shrink
                    #// axis (now in dense form) so we can ignore dense->end below.
                    if (shrink_axis_mask_inp & (1 << i)):
                        final_shape_gather_indices.append(kShrinkAxis);
                        shrink_axis_mask |= (1 << full_index);
                    else:
                        final_shape_gather_indices.append(full_index);

                    full_index = full_index + 1;
            return begin_mask, end_mask, shrink_axis_mask

        def canonicalindices(sliceindex, stride_i, valid_range, masks, index, size):
            if (masks[index] != 0):
                return valid_range[index] if stride_i > 0 else valid_range[(index + 1) & 1]
            else:
                slice_fwd = (size + sliceindex) if sliceindex < 0 else sliceindex
                return valid_range[0] if slice_fwd < valid_range[0] else (valid_range[1] if slice_fwd > valid_range[1] else slice_fwd)

        splitinput = inputs[0]
        begin_mask, end_mask, shrink_axis_mask = transform_ellipsis()
        #print("begin_mask, end_mask, shrink_axis_mask", begin_mask, end_mask, shrink_axis_mask)
        sizes = []
        #print("input", begin, end, stride)
        for i in range(len(inputs_0_shape[0])):
            #print("begin ", begin[i])
            masks = []
            end_range = inputs_0_shape[0][i];
            begin_range = 0;
            masks.append(begin_mask & (1 << i))
            masks.append(end_mask & (1 << i))
            #print("masks ", masks[0], masks[1])
            begin_and_end_masked = (begin_mask & (1 << i) != 0) and (end_mask & (1 << i) != 0)
            shrink_i = shrink_axis_mask & (1 << i)

            if (stride[i] < 0):
                end_range = inputs_0_shape[0][i] - 1;
                begin_range = -1;


            if (len(begin) != 0) and (len(end) != 0):
                if (shrink_i != 0):
                    slice_fwd = inputs_0_shape[0][i] + begin[i] if begin[i] < 0 else begin[i];
                    begin[i] = slice_fwd;
                    end[i] = slice_fwd + 1;
                else:
                    begin[i] = canonicalindices(begin[i], stride[i], [begin_range, end_range], masks, 0, inputs_0_shape[0][i])
                    end[i] = canonicalindices(end[i], stride[i], [begin_range, end_range], masks, 1, inputs_0_shape[0][i])

            '''print("i ", i)
            print("begin[i] ", begin[i])
            print("end[i] ", end[i])'''

            if (len(begin) != 0) and (len(end) != 0):
                interval = abs(end[i] - begin[i]);

            elif (shrink_i != 0):
                interval = 1;

            elif (begin_and_end_masked):
                interval = inputs_0_shape[0][i];

            if (stride[i] < 0):
                begin[i] = inputs_0_shape[0][i] - begin[i] - 1;
                end[i] = inputs_0_shape[0][i] - end[i] - 1;
                stride[i] = -stride[i];
                splitinput = _sym.reverse(splitinput, axis=i)
                #print("reveresed ", splitinput)
            #print("interval ", interval, "stride[i] ", stride[i])
            sliceinp = int((interval / stride[i]) + (1 if (interval % stride[i]) != 0 else 0))
            sizes.append(sliceinp)

            # if all the elements in the axis are to be sliced
            if (begin[i] == 0) and (end[i] == inputs_0_shape[0][i] or end[i] == end_range) and (stride[i] == 1):
                out = splitinput
                continue

            remainelem = inputs_0_shape[0][i] - begin[i]
            #print("remainelem ", remainelem)
            #end[i] = end[i] + 1 if (end[i] == end_range) and (end[i] != remainelem) else end[i]
            #print("begin[i] ", begin[i])
            #print("end[i] ", end[i])
            #sliceinp = int((max(end[i], inputs_0_shape[0][i]) - begin[i]) / stride[i])
            #interval = max(end[i], begin[i]) - begin[i]

            #print("sliceinp ", sliceinp)
            #sliceinp = 1 if sliceinp < 1 else sliceinp
            #print("sliceinp after:", sliceinp) 
            assert(sliceinp <= remainelem), "Input is wrong!!!"
            splitindex = 1
            if (begin[i] == 0):
                indices_or_sections = [begin[i] + sliceinp]
                splitindex = 0
            elif (sliceinp == remainelem):
                indices_or_sections = [begin[i]]
            else:
                indices_or_sections = [begin[i], begin[i] + sliceinp]

            splitoutput = _sym.split(splitinput, indices_or_sections=indices_or_sections, axis=i)
            splitinput = splitoutput[splitindex]
            out = splitoutput[splitindex]
        #split1 = _sym.split(inputs[0], indices_or_sections = [1, 2], axis=0)
        #split2 = _sym.split(split1[1], indices_or_sections = [1], axis=1)
        #return split2[1]
        final_shape = []
        axes = []
        axis_i = 0
        for index in final_shape_gather_indices:
            if (index >= 0):
                final_shape.append(sizes[index]);
            elif (index == kNewAxis):
                final_shape.append(1);
                axes.append(axis_i)
            axis_i = axis_i + 1
        #print("final_shape ", final_shape, " sizes ", sizes)
        #outshape = tuple(final_shape)
        #print("outshape ", outshape)
        #print("axes ", axes)
        for axisindex in axes:
            i = 0
            out = _sym.expand_dims(out, axis=axisindex + i)
            i = 1
        return out
    return _impl

def _stridedSlice():
    def _impl(inputs, attr, params):
        new_inputs = [[]]
        #print("input", params[inputs[1].list_output_names()[0]].asnumpy().size)
        begin = [params[inputs[1].list_output_names()[0]].asnumpy()[i] for i in range(params[inputs[1].list_output_names()[0]].asnumpy().size)]
        end = [params[inputs[2].list_output_names()[0]].asnumpy()[i] for i in range(params[inputs[2].list_output_names()[0]].asnumpy().size)]
        stride = [params[inputs[3].list_output_names()[0]].asnumpy()[i] for i in range(params[inputs[3].list_output_names()[0]].asnumpy().size)]
        #print("input", begin, end, stride)
        new_inputs[0] = inputs[0]
        return AttrCvt(
                op_name="strided_slice",
                extras={'begin':begin, 'end':end, 'stride':stride},
                #extras={'fill_value':3.0, 'dtype':'int32'},
                ignores=['ellipsis_mask', 'index_type', 'T', '_output_shapes', 'axis', 'N', 'shrink_axis_mask', 'Index', 'end_mask', 'new_axis_mask', 'begin_mask'])(new_inputs, attr)
        '''begin = inputs.pop(1)
        end = inputs.pop(2)
        stride = inputs.pop(3)
        return get_nnvm_op("strided_slice")(*inputs, begin = begin, end = end, stride = stride)'''
    return _impl
# compatible operators that do NOT require any conversion.
_identity_list = []

# _convert_map defines maps of name to converter functor(callable)
# for 1 to 1 mapping, use Renamer if nothing but name is different
# use AttrCvt if attributes need to be converted
# for 1 to N mapping(composed), use custom callable functions
# for N to 1 mapping, currently not supported(?)
_convert_map = {
    'AvgPool'                           : _pooling('avg_pool'),
    'BatchNormWithGlobalNormalization'  : _batch_norm(),
    'BiasAdd'                           : _bias_add(),
    'Cast'                              : _cast(),
    'CheckNumerics'                     : _check_numerics(),
    'Concat'                            : _concat(),
    'ConcatV2'                          : _concatV2(),
    'Conv2D'                            : _conv(),
    'DecodeJpeg'                        : _decode_image(),
    'ExpandDims'                        : _expand_dims(),
    'Identity'                          : _identity(),
    'MatMul'                            : _matmul(),
    'MaxPool'                           : _pooling('max_pool'),
    'Mul'                               : _elemwise('mul'),
    'Relu'                              : AttrCvt('relu'),
    'Reshape'                           : _reshape(),
    'ResizeBilinear'                    : _resize_bilinear(),
    'Softmax'                           : AttrCvt('softmax', {'axis': ('axis', 1)}),
    'Sub'                               : _elemwise('sub'),

    'Add'                               : _elemwise('add'),
    'Fill'                              : _fill(),
    'Gather'                            : _Gather(),
    'Stack'                              : _pack(),
    'Pack'                              : _pack(),
    'StridedSlice'                      : _stridedSlice(),
}

_convert_map_rnn = {
    'LSTMBlockCell'                     : _LSTMBlockCellWrapper(),
    }

class GraphProto(object):
    """ A helper class for handling nnvm graph copying from Tensorflow GraphDef.
    Definition:
        https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/framework/graph.proto
    """
    def __init__(self):
        self._nodes = {}
        self._params = {}
        self._renames = {}
        self._replacements = {}
        self._output_shapes = {}
        self._num_input = 0
        self._num_param = 0
        self._input_node = ''
        self._num_rnn_layer = 0
        self._out_states = []

    def from_tensorflow(self, graph):
        """Construct nnvm nodes from tensor flow  graph definition - GraphDef.

        Follow the tensorflow graph definition to parse and convert it to NNVM.
        Some of the assumptions listed below.

            -> First Const node will be comsidered as graph input.
            -> Rest all Const nodes are params.
            -> Last node is assumed as graph output.
            -> _output_shapes : Attribute should present in the tenserflow forzen graph.
            -> DecodeJpeg, ResizeBilinear: These are dymmy operators.
                                           Hence user should handle preprocessing outside.
            -> CheckNumerics: No implementation as of now for this.
                              Just copies input to output.


        Parameters
        ----------
        graph : tensorflow graph definition object
            The loaded tensorflow GraphDef

        Returns
        -------
        sym : nnvm.sym.Symbol
            The returned nnvm symbol
        params : dict
            A dict of name: tvm.nd.array pairs, used as pretrained weights
        """
        # Parse throught all nodes and start extracting
        # params aka Const nodes
        # input nodes  : First const node
        # normal nodes : other normal nodes

        try:
            from tensorflow.python.framework import tensor_util
        except ImportError as e:
            raise ImportError(
                "Unable to import tensorflow which is required {}".format(e))

        for node in graph.node:
            # Tensor flow doesn't have seperate list for params extraction.
            # Operator name 'Const' is treated as a parameter to build NNVM params dict.
            attr = self._parse_attr(node.attr)
            input_shapes = {}
            if node.op == "Placeholder":
                # Assuming only one input graph with type 'Placeholder'
                self._input_node = node.name
                self._num_input += 1
                self._nodes[node.name] = _sym.Variable(name=node.name)
                self._output_shapes[node.name] = \
                     [tensor_util.TensorShapeProtoToList(shape) for shape in attr['_output_shapes']]
                input_shapes[self._nodes[node.name]] = self._output_shapes[node.name]
                attr['_input_shapes'] = input_shapes
            elif node.op == "Const":
                # Assuming first Const node as Graph Input node
                if self._input_node == '':
                    self._input_node = node.name
                    self._num_input += 1
                    self._nodes[node.name] = _sym.Variable(name=node.name)
                else:
                    # Rest all nodes are Param nodes, lets parse
                    self._num_param += 1
                    for key, value in node.attr.items():
                        self._parse_param(key, value, node.name)
                    if node.name not in self._nodes:
                        raise NotImplementedError( \
                            "Const {} couldn't be converted to Param.".format(node.name))
            else:
                self._output_shapes[node.name] = \
                     [tensor_util.TensorShapeProtoToList(shape) for shape in attr['_output_shapes']]

                try:
                    """ToDo: Some of the tensorflow operators internaly maintain layers and
                    its output name will the layer number along with graph node name.
                    eg: Node name:- 'Model/RNN/cell_0/RnnCell', but the output name
                    will be 'Model/RNN/cell_0/RnnCell:0'
                    In thin case, the digit has to be ignored
                    """
                    if ":" in node.input[0]:
                        in_name, _ = node.input[0].split(':')
                        node.input[0] = in_name

                    inputs = [self._nodes[i] for i in node.input]
                    for i in node.input:
                        if i not in self._params:
                            input_shapes[self._nodes[i]] = self._output_shapes[i]
                    attr['_input_shapes'] = input_shapes
                except KeyError:
                    # TODO: Need to find clean way to handle '^CheckNumerics'
                    print ("Some Exception while inputs list:", node.input, " ignoring...")

                inputs = self._fix_extranodes(node.op, attr, inputs)

                op = self._convert_operator(node.op, inputs, attr, graph)
                # Assuming only one output.
                self._nodes[node.name] = op
                node_output = op
        # Assume the final node is the output node
        out = node_output
        #Set the RNN states as another output if exists
        if (self._num_rnn_layer > 0):
            out_states = _sym.concatenate(*self._out_states, axis=0)
            out = [out, out_states]

        if isinstance(out, list):
            out = _sym.Group(out)
        return out, self._params

    def _parse_param(self, key, value, name):
        try:
            from tensorflow.python.framework import tensor_util
        except ImportError as e:
            raise ImportError(
                "Unable to import tensorflow which is required {}".format(e))

        if key == 'value':
            np_array = tensor_util.MakeNdarray(value.tensor)
            array_ndim = len(np_array.shape)
            if array_ndim == 0:
                new_array = np.empty([1], dtype=np_array.dtype)
                new_array[0] = np_array
                self._params[name] = tvm.nd.array(new_array)
            else:
                self._params[name] = tvm.nd.array(np_array)
            self._nodes[name] = _sym.Variable(name=name,
                                              shape=self._params[name].shape)
        else:
            if key != 'dtype' and key != '_output_shapes':
                raise NotImplementedError \
                    ("Other attributes for a Const(param) Node {} ? .".format(key))

    def _get_attr(self, buf):
        """Returns the value of the attr of this buf with the given `name`.

        Args:
          buf: attrvalue protobuf.

        Returns:
          The value of the attr, as a Python object.

        Raises:
          ValueError: If this op does not have an attr with the given `name`.
        """
        fields = ["s", "i", "f", "b", "type", "shape", "tensor", "func"]

        x = buf

        ret = []

        try:
            from tensorflow.python.framework import dtypes
        except ImportError as e:
            raise ImportError(
                "Unable to import tensorflow which is required {}".format(e))

        # Treat an empty oneof value as an empty list.
        if not x.WhichOneof("value"):
            return ret
        if x.HasField("list"):
            for f in fields:
                if getattr(x.list, f):
                    if f == "type":
                        ret = [dtypes.as_dtype(x) for x in list(getattr(x.list, f))]
                    else:
                        ret = list(getattr(x.list, f))
        else:
            for f in fields:
                if x.HasField(f):
                    if f == "type":
                        ret = dtypes.as_dtype(getattr(x, f))
                    else:
                        ret = getattr(x, f)
        return ret

    def _parse_attr(self, attr_proto):
        """Convert a list of AttributeProto to a dict, with names as keys."""
        attrs = {}
        for key, value in attr_proto.items():
            attrs[key] = self._get_attr(value)

        return attrs

    def _convert_rnn_operator(self, op_name, inputs, attrs, _params, graph, convert_map=None):
        """Convert RNN and its variant operators to NNVM operators.
        This converter read the input states of each layers and
        also maintain the output states of each layer in a list.

        Parameters
        ----------
        op_name : str
            Operator name, such as Conv2D, AvgPool
        inputs : list of nnvm.Symbol
            List of input symbols.
        attrs : dict
            Dict of operator attributes
        params : dict
            List of  pretrained weights and bias
        graph : Tensorflow graph object
            Graph is to find the number of upcoming same operator to
            calculate the number of layers.
        convert_map : dict
            Dict of name : callable, where name is the op's name that
            require conversion to nnvm, callable are functions which
            take attrs and return (new_op_name, new_attrs)

        Returns
        -------
        sym : nnvm.Symbol
            Converted nnvm Symbol
        """
        in_state_c_name = op_name+"_param_in_state_c"
        in_state_h_name = op_name+"_param_in_state_h"
        if self._num_rnn_layer == 0:
            """ToDo: Dummy symbol is created in case the graph node is the first layer
            in the graph. The initial state is created inside the operator wrapper function
            because the operator specific diamentions and state's shapes has to be identified"""
            in_state_c = _sym.Variable("")
            in_state_h = _sym.Variable("")
        else:
            in_state_c = self._nodes[in_state_c_name]
            in_state_h = self._nodes[in_state_h_name]
        sym, cur_out_state, in_state_c, in_state_h = convert_map[op_name](op_name, inputs,
                                                                          in_state_c_name,
                                                                          in_state_h_name,
                                                                          in_state_c, in_state_h,
                                                                          attrs, _params,
                                                                          graph,
                                                                          self._num_rnn_layer)
        self._nodes[in_state_c_name] = in_state_c
        self._nodes[in_state_h_name] = in_state_h
        cur_out_state = _sym.expand_dims(cur_out_state, axis=0, num_newaxis=1)
        self._out_states.append(cur_out_state)
        self._num_rnn_layer += 1
        return sym

    def _convert_operator(self, op_name, inputs, attrs, graph, identity_list=None, convert_map=None):
        """Convert from Tensorflow operator to nnvm operator.
        The converter must specify conversions explicity for incompatible name, and
        apply handlers to operator attributes.

        Parameters
        ----------
        op_name : str
            Operator name, such as Conv2D, AvgPool
        inputs : list of nnvm.Symbol
            List of input symbols.
        attrs : dict
            Dict of operator attributes
        graph : Tensorflow graph object
            Graph is to find the number of upcoming same node to
            calculate the number of layers.
        identity_list : list
            List of operators that don't require conversion
        convert_map : dict
            Dict of name : callable, where name is the op's name that
            require conversion to nnvm, callable are functions which
            take attrs and return (new_op_name, new_attrs)

        Returns
        -------
        sym : nnvm.Symbol
            Converted nnvm Symbol
        """
        identity_list = identity_list if identity_list else _identity_list
        convert_map = convert_map if convert_map else _convert_map
        convert_map_rnn = _convert_map_rnn
        if op_name in identity_list:
            sym = get_nnvm_op(op_name)(*inputs, **attrs)
        elif op_name in convert_map:
            sym = convert_map[op_name](inputs, attrs, self._params)
        elif op_name in convert_map_rnn:
            sym = self._convert_rnn_operator(op_name, inputs, attrs,
                                         self._params, graph, convert_map_rnn)
        else:
            raise NotImplementedError("Operator {} not implemented.".format(op_name))
        return sym

    def _fix_extranodes(self, op_name, attr, inputs):
        if op_name == "Softmax":
            # Require some times flatten of data before it goes to softmax
            # Need to relook into this with latest softmax axis support.
            op = AttrCvt(op_name='flatten')(inputs, {})
            node_output = op.list_output_names()
            for k, i in zip(list(node_output), range(len(node_output))):
                self._nodes[k] = op[i]
            inputs = [op]

        return inputs

def from_tensorflow(graph):
    """  Load tensorflow graph which is a python tensorflow graph object into nnvm graph.
    The companion parameters will be handled automatically.

    Parameters
    ----------
    graph : GraphDef object
        Tensorflow GraphDef

    Returns
    -------
    sym : nnvm.Symbol
        Compatible nnvm symbol

    params : dict of str to tvm.ndarray
        Dict of converted parameters stored in tvm.ndarray format
    """
    g = GraphProto()
    sym, params = g.from_tensorflow(graph)
    return sym, params

# pylint: disable=invalid-name, unused-argument
"""Quantization operators"""
from __future__ import absolute_import

import tvm
import topi
from topi.util import get_const_int
from . import registry as reg


def _fschedule_naive(_, outs, target):
    return tvm.create_schedule([x.op for x in outs])


@reg.register_compute("quantize")
def compute_quantize(attrs, inputs, _):
    """Compute definition of quantize"""
    repr_bit = attrs.get_int('repr_bit')
    out_dtype = attrs['out_type']
    assert out_dtype == 'int8'
    data = inputs[0]

    bit_width = 8
    repr_value = pow(2, repr_bit)
    limit = (pow(2, bit_width - 1) - 1) * repr_value
    cliped_data = topi.clip(data, -limit, limit)
    scaled_data = tvm.compute(data.shape, lambda *i: \
        cliped_data(*i) / repr_value)
    round_data = tvm.compute(data.shape, lambda *i: \
        tvm.select(scaled_data(*i) < 0,
                   - (- scaled_data(*i) + 0.5),
                   scaled_data(*i) + 0.5))
    return topi.cast(round_data, out_dtype)


reg.register_schedule("quantize", _fschedule_naive)


@reg.register_compute("dequantize")
def compute_dequantize(attrs, inputs, _):
    """Compute definition of dequantize"""
    repr_bit = attrs.get_int('repr_bit')
    data = inputs[0]
    scaled_data = tvm.compute(data.shape, lambda *i: (data(*i)) * float(pow(2, repr_bit)))
    return scaled_data

reg.register_schedule("dequantize", _fschedule_naive)


@reg.register_compute("quantized_dense")
def compute_quantized_dense(attrs, inputs, _):
    """Compute definition of quantized_dense"""
    out_dtype = attrs['out_type']
    assert attrs.get_bool("use_bias") is False

    data = inputs[0]
    weight = inputs[1]
    m, l = data.shape
    n, _ = weight.shape

    k = tvm.reduce_axis((0, l), name='k')
    out = tvm.compute((m, n), lambda i, j: \
        tvm.sum(data[i][k].astype(out_dtype) *
                weight[j][k].astype(out_dtype), axis=k))
    return out

reg.register_schedule("quantized_dense", _fschedule_naive)


@reg.register_compute("quantized_conv2d")
def compute_quantized_conv2d(attrs, inputs, _):
    """Compute definition of quantized_conv2d"""
    padding = attrs.get_int_tuple("padding")
    strides = attrs.get_int_tuple("strides")
    dilation = attrs.get_int_tuple("dilation")
    groups = attrs.get_int("groups")
    channels = attrs.get_int("channels")
    layout = attrs["layout"]
    out_dtype = attrs['out_type']

    assert layout == "NCHW", "only support nchw for now"
    assert dilation == (1, 1), "not support dilate now"
    assert attrs.get_bool("use_bias") is False
    if groups == 1:
        out = topi.nn.conv2d(inputs[0],
                             inputs[1],
                             strides,
                             padding,
                             out_dtype=out_dtype)
    elif groups == get_const_int(inputs[0].shape[1]) and groups == channels:
        out = topi.nn.depthwise_conv2d_nchw(inputs[0],
                                            inputs[1],
                                            strides,
                                            padding,
                                            out_dtype=out_dtype)
    else:
        raise ValueError("not support arbitrary group number for now")
    return out

@reg.register_schedule("quantized_conv2d")
def schedule_quantized_conv2d(attrs, outs, target):
    groups = attrs.get_int("groups")
    with tvm.target.create(target):
        if groups == 1:
            return topi.generic.schedule_conv2d_nchw(outs)
        return topi.generic.schedule_depthwise_conv2d_nchw(outs)

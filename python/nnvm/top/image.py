# pylint: disable=invalid-name, unused-argument
"""Tensor Image Operations based on TVM extern interface"""
from __future__ import absolute_import

import tvm
import topi
from . import registry as reg
from .registry import OpPattern

def _schedule_extern(_, outs, target):
    """Generic schedule for image operations"""
    with tvm.target.create(target):
        return topi.generic.schedule_extern(outs)

_fschedule_extern = tvm.convert(_schedule_extern)

# image_decode
@reg.register_compute("image_decode")
def compute_image_decode(attrs, inputs, out_info):
    """Compute definition of image_decode"""

    shape_tuple = attrs.get_int_tuple("input_shape")
    channels = attrs.get_int("channels")
    oshape = (shape_tuple[0], shape_tuple[1], channels)
    out = tvm.extern(oshape, inputs,
                     lambda ins, outs:
                     tvm.call_packed("tvm.contrib.image.decode",
                                     ins[0], outs[0]), dtype='uint8', name="image_output")
    return out

reg.register_pattern("image_decode", OpPattern.OPAQUE)
reg.register_schedule("image_decode", _fschedule_extern)

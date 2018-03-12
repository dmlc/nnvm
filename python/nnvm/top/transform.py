# pylint: disable=invalid-name, unused-argument
"""Tensor transformation ops"""
from __future__ import absolute_import

import topi
from .tensor import _fschedule_broadcast, _fschedule_injective
from . import registry as reg
from .registry import OpPattern

# expand_dims
reg.register_pattern("expand_dims", OpPattern.BROADCAST)
reg.register_schedule("expand_dims", _fschedule_broadcast)

# expand_like
@reg.register_compute("expand_like")
def compute_reshape_like(attrs, inputs, _):
    """Compute definition of expand_like"""
    exclude = attrs.get_bool("exclude")
    axis = attrs.get_int_tuple("axis")
    return topi.transform.expand_like(inputs[0], inputs[1], axis, exclude)
reg.register_pattern("expand_like", OpPattern.BROADCAST)
reg.register_schedule("expand_like", _fschedule_broadcast)

# transpose
reg.register_pattern("transpose", OpPattern.INJECTIVE)
reg.register_schedule("transpose", _fschedule_injective)

# reshape
reg.register_pattern("reshape", OpPattern.INJECTIVE)
reg.register_schedule("reshape", _fschedule_injective)

# reshape_like
@reg.register_compute("reshape_like")
def compute_reshape_like(_, inputs, _):
    """Compute definition of reshape_like"""
    return topi.reshape(inputs[0], inputs[1].shape)
reg.register_pattern("reshape_like", OpPattern.INJECTIVE)
reg.register_schedule("reshape_like", _fschedule_injective)

# squeeze
reg.register_pattern("squeeze", OpPattern.INJECTIVE)
reg.register_schedule("squeeze", _fschedule_injective)

# concatenate
reg.register_pattern("concatenate", OpPattern.INJECTIVE)
reg.register_schedule("concatenate", _fschedule_injective)

# split
reg.register_pattern("split", OpPattern.INJECTIVE)
reg.register_schedule("split", _fschedule_injective)

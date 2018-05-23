# pylint: disable=invalid-name, unused-argument
"""Reduction ops"""
from __future__ import absolute_import

import tvm
import topi
import topi.cuda

def _schedule_reduce(_, outs, target):
    """Generic schedule for reduce"""
    with tvm.target.create(target):
        return topi.generic.schedule_reduce(outs)


_fschedule_reduce = tvm.convert(_schedule_reduce)

def _compute_reduce(f):
    """auxiliary function"""
    def _compute(attrs, inputs, out_info):
        axis = attrs.get_int_tuple("axis")
        keepdims = attrs.get_bool("keepdims")
        if axis:
            return f(inputs[0], axis=axis, keepdims=keepdims)
        return f(inputs[0], keepdims=keepdims)
    return _compute

# coding: utf-8
from __future__ import absolute_import
import numpy as np
import scipy
import scipy.stats
import math

import tvm
from tvm.contrib import graph_runtime
from collections import namedtuple

from . import graph as _graph
from . import compiler as _compiler
from .compiler import graph_attr
from .compiler.graph_util import infer_shape, infer_dtype
from .compiler.build_module import precompute_prune

_collect_internal_outputs = tvm.get_global_func("nnvm.quantization.CollectInternalOutputs")

CalibrationEntry = namedtuple("CalibrationEntry", ['min_value', 'max_value'])

def execute_graph(module, inputs, oshapes, odtypes):
    module.set_input(**inputs)
    module.run()

    outs = []
    for i in range(len(oshapes)):
        arr = tvm.nd.empty(oshapes[i], dtype=odtypes[i])
        module.get_output(i, arr)
        outs.append(arr)

    return outs

def _shape_dtype_dict(inputs, params=None):
    ishapes = {k : v.shape for k, v in inputs.items()}
    idtypes = {k : v.dtype for k, v in inputs.items()}
    if params is not None:
        for key, param in params.items():
            ishapes[key] = param.shape
            idtypes[key] = param.dtype
    return ishapes, idtypes


def collect_statistics(graph, dataset, params={}):
    ishapes, idtypes = _shape_dtype_dict(dataset[0], params)

    # optimize
    graph = graph.apply('SeparateBias')
    graph = graph_attr.set_shape_inputs(graph, ishapes)
    graph = graph.apply(["InferShape", "SimplifyInference"])
    graph = graph_attr.set_shape_inputs(graph, ishapes)
    graph = graph.apply(["InferShape", "FoldScaleAxis"])
    graph, params = precompute_prune(graph, params)
    ishapes, idtypes = _shape_dtype_dict(dataset[0], params)

    # transform to statistic graph
    stats_graph = _collect_internal_outputs(graph);

    # build module
    stats_graph, lib, _ = _compiler.build(stats_graph.symbol, "llvm", ishapes, idtypes)
    m = graph_runtime.create(stats_graph, lib, tvm.cpu(0))
    m.set_input(**params)

    # execute and collect stats
    records = {}  # dict from node name to list of entry
    out_names = stats_graph.symbol.list_output_names()
    _, oshapes = infer_shape(stats_graph, **ishapes)
    _, odtypes = infer_dtype(stats_graph, **idtypes)
    for inputs in dataset:
        outs = execute_graph(m, inputs, oshapes, odtypes)
        for i, out in enumerate(outs):
            key = out_names[i]
            min_value = np.amin(out.asnumpy())
            max_value = np.amax(out.asnumpy())
            entry = {'min_value': min_value, 'max_value': max_value}
            if key in records:
                records[key].append(entry)
            else:
                records[key] = [entry]

    # analysis
    # print('records:')
    base2_range = []
    for name in out_names:
        # print('{}:'.format(name))
        # for entry in records[name]:
        #     print("{}, {}".format(entry['min_value'], entry['max_value']))
        lower_bound = min(entry['min_value'] for entry in records[name])
        upper_bound = max(entry['max_value'] for entry in records[name])
        eps = pow(2, -30)
        k0 = int(math.ceil(math.log(abs(lower_bound) + eps, 2)))
        k1 = int(math.ceil(math.log(abs(upper_bound) + eps, 2)))
        base2_range.append(max(k0, k1))

    graph._set_json_attr("base2_range", base2_range, "list_int")
    return graph, params

def quantize(graph, debug=False):
    graph._set_json_attr("debug", int(debug), "int")
    qgraph = graph.apply("Quantize")
    return qgraph

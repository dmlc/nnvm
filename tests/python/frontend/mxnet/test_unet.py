import mxnet.ndarray as nd
import mxnet as mx
import numpy as np
import tvm

sym_mx = mx.sym.load("model_zoo/unet.json")
import nnvm
sym, params = nnvm.frontend.from_mxnet(sym_mx)
import nnvm.compiler

# target = 'llvm'
# x = np.zeros((16, 32, 256, 256))
# shape_dict = {'data': x.shape}
# graph, lib, params = nnvm.compiler.build(sym, target, shape_dict, params=params)

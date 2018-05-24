import sys
import nnvm.frontend.mxnet
import mxnet as mx
import numpy as np
import nnvm.compiler
import tvm
from tvm.contrib import graph_runtime

target = 'llvm'
target_to_device={'opencl':tvm.cl(0), 'llvm':tvm.cpu(0), 'cuda':tvm.gpu(0), }
ctx = target_to_device[target]
dtype = 'float32'

def test_clip():
    x = np.arange(-1,8)
    var = mx.sym.var('data')
    mx_sym = mx.sym.clip(data=var, a_min=0, a_max=6, name='clip')
    args, auxs = [], []
    nnvm_sym, nnvm_params = nnvm.frontend.from_mxnet(mx_sym, args, auxs)
    shape_dict = {'data': x.shape}
    graph, lib, params = nnvm.compiler.build(nnvm_sym, target, shape_dict, params=nnvm_params)
    m = graph_runtime.create(graph, lib, ctx)
    m.set_input(**params)
    m.set_input('data', tvm.nd.array(x.astype(dtype)))
    m.run()
    tvm_output = m.get_output(0, tvm.nd.empty(x.shape, dtype)).asnumpy()
    assert(tvm_output[0]==0)
    assert(tvm_output[-1]==6)

def test_concat():
    a, b = np.zeros((2,3)), np.ones((2,3))
    var0, var1 = mx.sym.var('data0'), mx.sym.var('data1')
    mx_sym = mx.sym.concat(var0, var1, dim=0)
    args, auxs = [], []
    nnvm_sym, nnvm_params = nnvm.frontend.from_mxnet(mx_sym, args, auxs)
    shape_dict = {'data0': (2,3),'data1': (2,3),}
    graph, lib, params = nnvm.compiler.build(nnvm_sym, target, shape_dict, params=nnvm_params)
    m = graph_runtime.create(graph, lib, ctx)
    m.set_input(**params)
    m.set_input('data0', tvm.nd.array(a.astype(dtype)))
    m.set_input('data1', tvm.nd.array(b.astype(dtype)))
    m.run()
    tvm_output = m.get_output(0, tvm.nd.empty((4,3), dtype)).asnumpy()
    assert np.allclose(tvm_output, np.vstack((a,b)))

def test_zeros_like():
    a = np.zeros((2,3))
    var0 = mx.sym.zeros(shape=(2,3))
    mx_sym = mx.sym.zeros_like(var0)
    args, auxs = [], []
    nnvm_sym, nnvm_params = nnvm.frontend.from_mxnet(mx_sym, args, auxs)
    shape_dict = {}
    graph, lib, params = nnvm.compiler.build(nnvm_sym, target, shape_dict, params=nnvm_params)
    m = graph_runtime.create(graph, lib, ctx)
    m.run()
    tvm_output = m.get_output(0, tvm.nd.empty((2,3), dtype)).asnumpy()
    assert np.allclose(tvm_output, a)

def test_ones_like():
    a = np.ones((2,3))
    var0 = mx.sym.ones(shape=(2,3))
    mx_sym = mx.sym.ones_like(var0)
    args, auxs = [], []
    nnvm_sym, nnvm_params = nnvm.frontend.from_mxnet(mx_sym, args, auxs)
    shape_dict = {}
    graph, lib, params = nnvm.compiler.build(nnvm_sym, target, shape_dict, params=nnvm_params)
    m = graph_runtime.create(graph, lib, ctx)
    m.run()
    tvm_output = m.get_output(0, tvm.nd.empty((2,3), dtype)).asnumpy()
    assert np.allclose(tvm_output, a)

def test_greater():
    a, b = np.array([[1,0,-1],[1,0,-1]]), np.zeros((2,3))
    var0, var1 = mx.sym.var('data0'), mx.sym.var('data1')
    mx_sym = mx.sym.broadcast_greater(var0, var1, name='greater')
    args, auxs = [], []
    nnvm_sym, nnvm_params = nnvm.frontend.from_mxnet(mx_sym, args, auxs)
    shape_dict = {'data0': (2,3),'data1': (2,3),}
    graph, lib, params = nnvm.compiler.build(nnvm_sym, target, shape_dict, params=nnvm_params)
    m = graph_runtime.create(graph, lib, ctx)
    m.set_input(**params)
    m.set_input('data0', tvm.nd.array(a.astype(dtype)))
    m.set_input('data1', tvm.nd.array(b.astype(dtype)))
    m.run()
    tvm_output = m.get_output(0, tvm.nd.empty((2,3), dtype)).asnumpy()
    assert np.allclose(tvm_output, np.array([[1,0,0],[1,0,0]]))

def test_lesser():
    a, b = np.array([[1,0,-1],[1,0,-1]]), np.zeros((2,3))
    var0, var1 = mx.sym.var('data0'), mx.sym.var('data1')
    mx_sym = mx.sym.broadcast_lesser(var0, var1, name='lesser')
    args, auxs = [], []
    nnvm_sym, nnvm_params = nnvm.frontend.from_mxnet(mx_sym, args, auxs)
    shape_dict = {'data0': (2,3),'data1': (2,3),}
    graph, lib, params = nnvm.compiler.build(nnvm_sym, target, shape_dict, params=nnvm_params)
    m = graph_runtime.create(graph, lib, ctx)
    m.set_input(**params)
    m.set_input('data0', tvm.nd.array(a.astype(dtype)))
    m.set_input('data1', tvm.nd.array(b.astype(dtype)))
    m.run()
    tvm_output = m.get_output(0, tvm.nd.empty((2,3), dtype)).asnumpy()
    assert np.allclose(tvm_output, np.array([[0,0,1],[0,0,1]]))

def test_leaky_relu():
    x = np.arange(-1,8)
    var = mx.sym.var('data')
    mx_sym = mx.sym.LeakyReLU(data=var, act_type='leaky', slope=.25, name='leaky_relu')
    args, auxs = [], []
    nnvm_sym, nnvm_params = nnvm.frontend.from_mxnet(mx_sym, args, auxs)
    shape_dict = {'data': x.shape}
    graph, lib, params = nnvm.compiler.build(nnvm_sym, target, shape_dict, params=nnvm_params)
    m = graph_runtime.create(graph, lib, ctx)
    m.set_input(**params)
    m.set_input('data', tvm.nd.array(x.astype(dtype)))
    m.run()
    tvm_output = m.get_output(0, tvm.nd.empty(x.shape, dtype)).asnumpy()
    assert(abs(tvm_output[0]+0.25)<1e-5)
    assert(tvm_output[-1]==7)
    
if __name__=='__main__':
    test_clip()
    test_concat()
    test_ones_like()
    test_zeros_like()
    test_greater()
    test_lesser()
    test_leaky_relu()

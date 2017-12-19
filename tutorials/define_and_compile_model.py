"""
Quick Start - Compile Resnet-18 Model for Raspberry Pi
===========================================
**Author**: `Yao Wang <https://github.com/kevinthesun>`_

This example shows how to build a neural network with NNVM python frontend and generate runtime library for Raspberry Pi with TVM.
To run this notebook, you need to install tvm and nnvm following `these instructions <https://github.com/dmlc/nnvm/blob/master/docs/how_to/install.md>`_.
Notice that you need to build tvm with llvm.

To begin with, we import NNVM and TVM.
"""
import tvm
import nnvm.compiler
import nnvm.testing


######################################################################
# Define Neural Network in NNVM
# -----------------------------
# First, let's define a neural network with nnvm python frontend.
# For simplicity, we'll use pre-defined resnet-18 network in NNVM.
# Parameters are initialized with Xavier initializer.
# NNVM also supports other model formats such as MXNet, CoreML and ONNX.
#
# In this tutorial, we assume we will do inference on our device
# and the batch size is set to be 1. Input images are RGB color
# images of size 224 * 224. We can call the :any:`nnvm.symbol.debug_str`
# to show the network structure.

batch_size = 1
image_shape = (3, 224, 224)
data_shape = (batch_size,) + image_shape

net, params = nnvm.testing.resnet.get_workload(batch_size=batch_size, image_shape=image_shape)
print(net.debug_str())

######################################################################
# Compile with TVM
# ----------------------------
# Next step is to compile model with tvm.
# The key parameter for compilation is optimization level.
# Currently this value can be 0 to 2, which corresponds to
# "SimplifyInference", "OpFusion" and "PrecomputePrune" respectively.
# In this example we set optimization level to be 0
# and use Raspberry Pi as compile target.
#
# By calling :any:`nnvm.compiler.build`, you will get three return values: graph,
# library and the new parameter, since we do some optimization that will
# change the parameters but keep the result of model same as original.

opt_level = 0
with nnvm.compiler.build_config(opt_level=opt_level):
    graph, lib, params = nnvm.compiler.build(
        net, tvm.target.rasp(), shape={"data": data_shape}, params=params)

######################################################################
# Save Compiled Module
# ----------------------------
# Finally, we can save the graph, lib and params into seperate files
# and deploy them to Raspberry Pi device.

from tvm.contrib import util

temp = util.tempdir()
path_lib = temp.relpath("deploy_lib.so")
lib.export_library(path_lib)
with open(temp.relpath("deploy_graph.json"), "w") as fo:
    fo.write(graph.json())
with open(temp.relpath("deploy_param.params"), "wb") as fo:
    fo.write(nnvm.compiler.save_param_dict(params))
print(temp.listdir())


"""
Compile Tensorflow Models
=====================
This article is a test script to test tensorflow models with NNVM.
All the required models and libraries will be downloaded from the internet
by the script.
"""

import tvm
import nnvm
import numpy as np
import tensorflow as tf
from tensorflow.python.ops import variable_scope
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import variables
from tensorflow.python import debug as tf_debug
from tensorflow.python.framework import graph_util

def init_data_array(ip_shape, dtype):
    shape_size = 1
    for posi in range (len(ip_shape)):
        shape_size = shape_size * ip_shape[posi]
    data_ary = np.arange(shape_size, dtype=dtype).reshape(ip_shape)
    return data_ary

def fill_data_array(ip_shape, ip_value, dtype):
    fill_ary = np.zeros(ip_shape, dtype=dtype)
    if isinstance(ip_value, int):
        fill_ary[0] = ip_value
    else:
        shape_size = 1
        for i in range(len(ip_shape)):
            shape_size = shape_size * ip_shape[i]
        fill_ary = fill_ary.flatten()
        ip_value_flatten = (np.asarray(ip_value)).flatten()
        for i in range(shape_size):
            fill_ary[i] = ip_value_flatten[i]
        fill_ary = fill_ary.reshape(ip_shape)
    return fill_ary

def test_forward(layer, inputs, ip_tensor, ip_name, dtype):
    '''Test network with given input image on both tensorflow and tvm'''

    def get_tensorflow_output(layer, ip_tensor, inputs):
        with tf.Session() as sess:
            feed_dict={}
            for i in range (len(ip_tensor)):
                feed_dict.setdefault(ip_tensor[i],inputs[i])
            tf_out = sess.run(layer, feed_dict)
            #tf.train.write_graph(sess.graph.as_graph_def(add_shapes=True), ".", "tf_reshape_test.pbtxt")
            graph_def = sess.graph.as_graph_def(add_shapes=True)
            #print(graph_def)
            return graph_def, tf_out

    def get_tvm_output(graph_def, ip_name, inputs, out_shape):
        '''Compute TVM output'''

        sym, params = nnvm.frontend.from_tensorflow(graph_def)

        target = 'llvm'
        shape_dict = {}
        for i in range (len(ip_name)):
            shape_dict.setdefault(ip_name[i],inputs[i].shape)
        dtype_dict = {}
        for i in range (len(ip_name)):
            dtype_dict.setdefault(ip_name[i],inputs[i].dtype)

        #with nnvm.compiler.build_config(opt_level=2):
        graph, library, params = nnvm.compiler.build(sym, target, shape_dict, dtype=dtype_dict, params=params)
        #print("graph:", graph.json())

        from tvm.contrib import graph_runtime
        ctx = tvm.cpu(0)
        m = graph_runtime.create(graph, library, ctx)

        # set inputs
        for i in range (len(ip_name)):
            m.set_input(ip_name[i], tvm.nd.array(inputs[i].astype(inputs[i].dtype)))

        m.set_input(**params)
        m.run()
        # get outputs
        tvm_out = m.get_output(0, tvm.nd.empty(out_shape, dtype)).asnumpy()
        return tvm_out

    graph_def, tf_output = get_tensorflow_output(layer, ip_tensor, inputs)
    #print("tf_output:", tf_output.shape, "\n", tf_output)
    tvm_out = get_tvm_output(graph_def, ip_name, inputs, tf_output.shape)
    #print("tvm_out:", tvm_out.shape, "\n", tvm_out)
    np.testing.assert_allclose(tf_output, tvm_out, rtol=1e-3, atol=1e-3)

def test_fill(ip_shape, fill_value, dtype):
    shape_length = 1
    for posi in range (len(ip_shape)):
        shape_length = shape_length * ip_shape[posi]
    tf.reset_default_graph()
    x = tf.placeholder(dtype, shape=ip_shape, name='x')
    y = tf.fill(ip_shape, fill_value)
    input_value = init_data_array(ip_shape, dtype)
    test_forward(y, [input_value], [x], [], dtype)

def test_stack(ip_shape, axis, num_ip, dtype):
    tf.reset_default_graph()
    input_names = [("input{:d}".format(i+1)) for i in range(num_ip)]
    inputs = [tf.placeholder(dtype, ip_shape, name=name) for name in input_names]
    output= tf.stack(inputs, axis=axis)
    input_values = [init_data_array(ip_shape, dtype) for name in input_names]
    test_forward(output, input_values, inputs, input_names, dtype)

def test_reshape(ip_shape, tf_op_shape, dtype):
    tf.reset_default_graph()
    x = tf.placeholder(dtype, ip_shape, name='x')
    y = tf.reshape(x, tf_op_shape, name="test_reshape")
    input_value = init_data_array(ip_shape, dtype)
    test_forward(y, [input_value], [x], ['x'], dtype)

def test_gather(ip_shape, indice_shape, indice_value, axis, dtype):
    tf.reset_default_graph()
    inputs = []
    input_names = []
    input_values = []
    params = tf.placeholder(dtype, ip_shape, name="x")
    inputs.append(params)
    input_names.append("x")
    indices = tf.placeholder("int32", indice_shape, name="indices")
    inputs.append(indices)
    input_names.append("indices")
    output= tf.gather(params, indices, axis=axis)
    input_values.append(init_data_array(ip_shape, dtype))
    input_values.append(fill_data_array(indice_shape, indice_value, "int32"))
    test_forward(output, input_values, inputs, input_names, dtype)

def test_lstm_cell_(batch_size, input_size, num_hidden,
            num_layers, num_times, forget_bias, dtype):
    tf.reset_default_graph()
    inputs = []
    input_names = []
    input_values = []

    ip_shape = (batch_size, input_size)
    state_shape = (batch_size, num_hidden)
    in_data = tf.placeholder(dtype, ip_shape, name="x")
    inputs.append(in_data)
    input_names.append("x")
    in_state_h = tf.placeholder("dtype", state_shape, name="state_h")
    in_state_c = tf.placeholder("dtype", state_shape, name="state_c")
    inputs.append(in_state_h)
    input_names.append("state_h")
    inputs.append(in_state_c)
    input_names.append("state_c")

def test_lstm_cell(batch_size, input_size, num_hidden,
            num_layers, num_times, forget_bias, dtype):
    tf.reset_default_graph()
    def get_tvm_output(graph_def):
        '''Compute TVM output'''
        sym, params = nnvm.frontend.from_tensorflow(graph_def)

        target = 'llvm'
        batch_size = 1
        num_hidden=2
        num_layers=1
        input_size = 2
        out_shape = (1, 2)
        out_state_shape=(2, 1, 2)
        shape_dict = {'root/Placeholder':(batch_size, input_size),
                      'LSTMBlockCell_param_in_state_c':(num_layers, batch_size, num_hidden),
                      'LSTMBlockCell_param_in_state_h':(num_layers, batch_size, num_hidden)}
        type_dict = {'LSTMBlockCell_param_in_state_c':'float32',
              'LSTMBlockCell_param_in_state_h':'float32'}
        graph, library, params = nnvm.compiler.build(sym, target, shape_dict,
                                                     type_dict, params=params)
        library.export_library("LSTM-test.so")
        with open("LSTM-test.json", "w") as fo:
            fo.write(graph.json())
        with open("LSTM-test.params", "wb") as fo:
            fo.write(nnvm.compiler.save_param_dict(params))

        from tvm.contrib import graph_runtime
        ctx = tvm.cpu(0)
        m = graph_runtime.create(graph, library, ctx)

        # set inputs
        input_data = np.full((num_layers, batch_size, num_hidden),1., dtype="float32")
        in_state_c = np.full((num_layers, batch_size, num_hidden),.1, dtype="float32")
        in_state_h = np.full((num_layers, batch_size, num_hidden),.1, dtype="float32")
        m.set_input('root/Placeholder', tvm.nd.array(input_data.astype("float32")))
        m.set_input('LSTMBlockCell_param_in_state_c', tvm.nd.array(in_state_c.astype("float32")))
        m.set_input('LSTMBlockCell_param_in_state_h', tvm.nd.array(in_state_h.astype("float32")))
        m.set_input(**params)
        m.run()
        # get outputs
        out = m.get_output(0, tvm.nd.empty(out_shape, dtype)).asnumpy()
        out_state = m.get_output(1, tvm.nd.empty(out_state_shape, dtype)).asnumpy()
        out_state_tup = np.split(out_state, indices_or_sections=2, axis=0)
        out_state_c = np.reshape(out_state_tup[0], (batch_size, num_hidden))
        out_state_h = np.reshape(out_state_tup[1], (batch_size, num_hidden))
        return [out, out_state_c, out_state_h]

    def get_tensorflow_output():
        with tf.Session() as sess:
            with variable_scope.variable_scope(
                "root", initializer=init_ops.constant_initializer(0.5)):
                m0 = array_ops.zeros([1, 2])
                m1 = array_ops.zeros([1, 2])
                #x = array_ops.ones([1, 2])
                x=tf.placeholder(shape=(1, 2), dtype='float32')
                print("Input name: ", x.name)
                g, ((out_m0, out_m1)) = \
                     tf.contrib.rnn.LSTMBlockCell(num_hidden,
                                                  forget_bias=forget_bias)(x, ((m0, m1)))

                sess.run([variables.global_variables_initializer()])
                res = sess.run([g, out_m0, out_m1], {
                    x.name: np.array([[1., 1.]]),
                    m0.name: 0.1 * np.ones([1, 2]),
                    m1.name: 0.1 * np.ones([1, 2]),
                })
            tf.train.write_graph(sess.graph.as_graph_def(add_shapes=True),
                                                         ".", "tf_lstm_test.pbtxt")
            graph_def = sess.graph.as_graph_def(add_shapes=True)
            final_graph_def = graph_util.convert_variables_to_constants(
                sess,
                graph_def,
                ['root/lstm_cell/LSTMBlockCell'],
                variable_names_whitelist=None,
                variable_names_blacklist=None)
            tf.train.write_graph(final_graph_def, ".", "tf_lstm_test_final.pbtxt")
            return final_graph_def, res

    graph_def, tf_out = get_tensorflow_output()
    tvm_out = get_tvm_output(graph_def)
    np.testing.assert_allclose(tf_out, tvm_out, rtol=1e-3, atol=1e-3)

#batch_size, input_size, num_hidden, num_layers, num_times, forget_bias
def test_forward_lstm():
    '''test LSTM block cell'''
    test_lstm_cell(1, 2, 2, 2, 2, 0.0, 'float32')

def test_forward_fill():
    '''test fill layer'''
    test_fill((10,), 7, 'int32')
    test_fill((3,4,5), 3.0, 'float32')
    test_fill((4,), 3.14, 'float32')
    test_fill((1,5,9), 3, 'int32')

def test_forward_reshape():
    '''test reshape layer'''
    test_reshape((4,), [2, -1], 'float32')
    test_reshape([1, 4], [2, -1], 'int32')
    test_reshape([1, 2, 3, 3], [2, -1], 'float32')
    test_reshape([1, 3, 2, 2], [-1, 3], 'int32')

def test_forward_stack():
    '''test stack layer'''
    test_stack((1,), 0, 1, 'float32')
    test_stack((1,), 0, 2, 'int32')
    test_stack((1,3), 0, 2, 'float32')
    test_stack((2,3), 1, 3, 'int32')
    test_stack((1,2,3), 0, 3, 'float32')
    test_stack((1,2,3), 3, 6, 'int32')

def test_forward_gather():
    '''test gather layer'''
    test_gather((4,), (1,), 1, 0, 'int32')
    test_gather((4,), (1,), 1, 0, 'float32')
    test_gather((1,4), (1,), [0], 0, 'int32')
    test_gather((4,), (1,2,2), [[1,0],[0,1]], 0, 'float32')
    test_gather((2,2), (1,2,2), [[1,0],[0,1]], 0, 'int32')
    test_gather((2,2), (1,2,2), [[1,0],[0,1]], 0, 'float32')
    test_gather((3,3,3), (1,1,2), [[1,0]], 0, 'int32')
    test_gather((4,3,5,6), (1,4), [[2,1,0,0]], 0, 'float32')

if __name__ == '__main__':
    test_forward_reshape()
    test_forward_stack()
    test_forward_fill()
    test_forward_gather()
    test_forward_lstm()

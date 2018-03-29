import copy
import numpy as np
import tvm
from tvm.contrib import graph_runtime
import topi.testing
import nnvm.symbol as sym
import nnvm.compiler
from nnvm.testing.config import ctx_list


def helper(symbol, inputs, dtype, np_forward, np_backward=None, **kwargs):
    input_shapes = {}
    input_symbols = []
    np_inputs = {}
    for (k, v) in inputs.items():
        input_shapes.update({k: v[0]})
        np_inputs.update({k: np.random.uniform(size=v[0]).astype(dtype)})
        if len(v) > 1:
            input_symbols.append(v[1])

    # put additional kwargs for np forward
    np_inputs_with_addtional_args = copy.copy(np_inputs)
    np_inputs_with_addtional_args.update(**kwargs)

    for target, ctx in ctx_list():
        graph, lib, _ = nnvm.compiler.build(symbol, target, input_shapes)
        module = graph_runtime.create(graph, lib, ctx)
        module.run(**np_inputs)
        y_np = np_forward(**np_inputs_with_addtional_args)
        out = module.get_output(0, tvm.nd.empty(y_np.shape, dtype))
        np.testing.assert_allclose(out.asnumpy(), y_np, atol=1e-5, rtol=1e-5)

        # backward
        if np_backward:
            graph._set_symbol_list_attr("grad_ys", symbol)
            graph._set_symbol_list_attr("grad_xs", input_symbols)
            graph._set_symbol_list_attr("grad_ys_out_grad", sym.Variable("head_grads"))
            graph = graph.apply("Gradient")
            input_shapes.update({"head_grads": y_np.shape})
            graph, lib, _ = nnvm.compiler.build(graph, target, input_shapes)
            module = graph_runtime.create(graph, lib, ctx)
            head_grads = np.random.uniform(size=y_np.shape).astype(dtype)
            np_inputs_with_head_gradients = copy.copy(np_inputs_with_addtional_args)
            np_inputs_with_head_gradients.update({'head_gradient': head_grads})
            np_gradients = np_backward(**np_inputs_with_head_gradients)
            if not isinstance(np_gradients, list):
                np_gradients = [np_gradients]
            module.run(head_grads=head_grads, **np_inputs)
            for i in range(len(np_gradients)):
                out = module.get_output(i, tvm.nd.empty(np_gradients[i].shape, dtype))
                np.testing.assert_allclose(out.asnumpy(), np_gradients[i], atol=1e-5, rtol=1e-5)


def test_relu():
    x = sym.Variable("x")
    y = sym.relu(sym.leaky_relu(x, alpha=0.3) - 0.2)

    def forward(x):
        x = (x < 0) * x * 0.3 + (x > 0) * x - 0.2
        return (x > 0) * x

    dtype = "float32"
    dshape = (1, 3, 32, 32)
    inputs = {'x': (dshape, x)}
    helper(y, inputs, dtype, forward)


def test_sym_scalar_pow():
    scalar = 3
    x = sym.Variable("x")
    y = x**scalar

    def forward(x):
        return x**scalar

    def backward(x, head_gradient):
        return scalar * x**(scalar - 1) * head_gradient

    dtype = "float32"
    dshape = (1, 3, 32, 32)
    inputs = {'x': (dshape, x)}
    helper(y, inputs, dtype, forward, backward)


def test_scalar_sym_pow():
    scalar = 3
    x = sym.Variable("x")
    y = scalar**x

    def forward(x):
        return scalar**x

    def backward(x, head_gradient):
        return np.log(scalar) * scalar**x * head_gradient

    dtype = "float32"
    dshape = (1, 3, 32, 32)
    inputs = {'x': (dshape, x)}
    helper(y, inputs, dtype, forward, backward)


def test_exp():
    x = sym.Variable("x")
    y = sym.exp(x)

    def forward(x):
        return np.exp(x)

    def backward(x, head_gradient):
        return np.exp(x) * head_gradient

    dtype = "float32"
    dshape = (1, 3, 32, 32)
    inputs = {'x': (dshape, x)}
    helper(y, inputs, dtype, forward, backward)


def test_log():
    x = sym.Variable("x")
    y = sym.log(x)

    def forward(x):
        return np.log(x)

    def backward(x, head_gradient):
        return 1. / x * head_gradient

    dtype = "float32"
    dshape = (1, 3, 32, 32)
    inputs = {'x': (dshape, x)}
    helper(y, inputs, dtype, forward, backward)


def test_tanh():
    x = sym.Variable("x")
    y = sym.tanh(x)

    def forward(x):
        return np.sinh(x) / np.cosh(x)

    def backward(x, head_gradient):
        y_np = forward(x)
        return (1 - y_np**2) * head_gradient

    dtype = "float32"
    dshape = (1, 3, 32, 32)
    inputs = {'x': (dshape, x)}
    helper(y, inputs, dtype, forward, backward)


def test_sigmoid():
    x = sym.Variable("x")
    y = sym.sigmoid(x)

    def forward(x):
        return 1.0 / (1.0 + np.exp(-x))

    def backward(x, head_gradient):
        y_np = forward(x)
        return y_np *(1 - y_np) * head_gradient

    dtype = "float32"
    dshape = (1, 3, 32, 32)
    inputs = {'x': (dshape, x)}
    helper(y, inputs, dtype, forward, backward)


def test_softmax():
    x = sym.Variable("x")
    y = sym.softmax(x)

    def forward(x):
        return topi.testing.softmax_python(x)

    dtype = "float32"
    dshape = (10, 1000)
    inputs = {'x': (dshape, x)}
    helper(y, inputs, dtype, forward)


def test_log_softmax():
    x = sym.Variable("x")
    y = sym.log_softmax(x)

    def forward(x):
        return topi.testing.log_softmax_python(x)

    dtype = "float32"
    dshape = (10, 1000)
    inputs = {'x': (dshape, x)}
    helper(y, inputs, dtype, forward)


def test_dense():
    x = sym.Variable("x")
    y = sym.dense(x, units=3, name="dense")
    y = sym.flatten(y)

    def forward(x, dense_weight, dense_bias):
        return np.dot(x, dense_weight.T) + dense_bias

    dtype = "float32"
    inputs = {
        'x': ((10, 100), x),
        'dense_weight': ((3, 100),),
        'dense_bias': ((3,),)
    }
    helper(y, inputs, dtype, forward)


def test_batchnorm():
    x = sym.Variable("x")
    beta = sym.Variable("beta")
    gamma = sym.Variable("gamma")
    moving_var = sym.Variable("moving_var")
    moving_mean = sym.Variable("moving_mean")
    eps = 1e-5
    y = sym.batch_norm(
        x, gamma, beta, moving_mean, moving_var, epsilon=eps)

    def forward(x, gamma, beta, moving_mean, moving_var):
        return (x - moving_mean) / np.sqrt(moving_var + eps) * gamma + beta

    dtype = "float32"
    inputs = {
        'x': ((10, 20), x),
        'gamma': ((20,),),
        'beta': ((20,),),
        'moving_mean': ((20,),),
        'moving_var': ((20,),)
    }

    helper(y, inputs,  dtype, forward)


def verify_concatenate(ishape, axis):
    x = [sym.Variable("x%d" % i) for i in range(len(ishape))]
    y = sym.concatenate(*x, axis=axis) + 1
    dtype = "float32"
    for target, ctx in ctx_list():
        # set input
        data = []
        for i, shape in enumerate(ishape):
            data.append(np.random.uniform(size=shape).astype(dtype))
        pdict = {"x%d" % i :  v for i, v in enumerate(data)}
        shape = {"x%d" % i :  v.shape for i, v in enumerate(data)}
        graph, lib, _ = nnvm.compiler.build(y, target, shape)
        m = graph_runtime.create(graph, lib, ctx)
        m.run(**pdict)
        out_np = np.concatenate(data, axis=axis) + 1
        out = m.get_output(0, tvm.nd.empty(out_np.shape))
        np.testing.assert_allclose(out.asnumpy(), out_np, atol=1e-5, rtol=1e-5)


def test_concatenate():
    verify_concatenate([(2, 3, 4), (1, 3, 4)], axis=0)
    verify_concatenate([(2, 4), (2, 7)], axis=1)


def verify_split(ishape, indices_or_sections, axis):
    x = sym.Variable("x")
    y = sym.split(x, indices_or_sections=indices_or_sections, axis=axis)
    dtype = "float32"
    x_np = np.random.uniform(size=ishape).astype(dtype)
    res = np.split(x_np, indices_or_sections, axis=axis)
    for target, ctx in ctx_list():
        # set input
        graph, lib, _ = nnvm.compiler.build(y, target, {"x": ishape})
        m = graph_runtime.create(graph, lib, ctx)
        m.run(x=x_np)
        for i, arr  in enumerate(res):
            out = m.get_output(i, tvm.nd.empty(arr.shape))
            np.testing.assert_allclose(out.asnumpy(), arr, atol=1e-5, rtol=1e-5)


def test_split():
    verify_split((2, 3), 2, axis=0)
    verify_split((5, 3), [3], axis=0)
    verify_split((5, 9, 3), [3, 4], axis=1)


def verify_squeeze(dshape, axis):
    x = sym.Variable("x")
    if axis:
        y = sym.squeeze(x, axis=axis)
    else:
        y = sym.squeeze(x)
    y = y + 1

    def forward(x):
        return np.squeeze(x, axis=axis) + 1

    dtype = "float32"
    inputs = {'x': (dshape, x)}
    helper(y, inputs, dtype, forward)


def test_squeeze():
    verify_squeeze((1, 3, 2, 5), None)
    verify_squeeze((1, 3, 1), axis=0)
    verify_squeeze((1, 3, 2, 5, 1), axis=-1)


def test_pad():
    x = sym.Variable("x")
    y = sym.pad(x, pad_width=((0, 0), (0, 0), (0, 1), (2, 3)), pad_value=1.)

    def forward(x):
        return np.pad(x,
                      pad_width=((0, 0), (0, 0), (0, 1), (2, 3)),
                      mode='constant', constant_values=1.)

    dtype = "float32"
    inputs = {'x': ((1, 3, 28, 28), x)}
    helper(y, inputs, dtype, forward)


def test_matmul():
    lhs = sym.Variable('lhs')
    rhs = sym.Variable('rhs')
    result = sym.matmul(lhs, rhs)

    def forward(lhs, rhs, transpose_a=False, transpose_b=False):
        if transpose_a:
            lhs = lhs.T
        if transpose_b:
            rhs = rhs.T
        return np.matmul(lhs, rhs)

    def backward(lhs, rhs, head_gradient, transpose_a=False, transpose_b=False):
        grad_a = None
        grad_b = None
        if not transpose_a and not transpose_b:
            grad_a = np.matmul(head_gradient, rhs.T)
            grad_b = np.matmul(lhs.T, head_gradient)
        elif transpose_a and not transpose_b:
            grad_a = np.matmul(rhs, head_gradient.T)
            grad_b = np.matmul(lhs, head_gradient)
        elif not transpose_a and transpose_b:
            grad_a = np.matmul(head_gradient, rhs)
            grad_b = np.matmul(head_gradient.T, lhs)
        else:
            grad_a = np.matmul(rhs.T, head_gradient.T)
            grad_b = np.matmul(head_gradient.T, lhs.T)
        return [grad_a, grad_b]

    dtype = "float32"
    inputs = {
        'lhs': ((1, 3), lhs),
        'rhs': ((3, 4), rhs),

    }
    helper(result, inputs, dtype, forward, backward)

    # with transpose
    lhs = sym.Variable('lhs')
    rhs = sym.Variable('rhs')
    result = sym.matmul(lhs, rhs, transpose_a=False, transpose_b=True)

    dtype = "float32"
    inputs = {
        'lhs': ((1, 3), lhs),
        'rhs': ((4, 3), rhs),
    }

    helper(result, inputs, dtype, forward, backward, transpose_b=True)


if __name__ == "__main__":
    test_split()
    test_concatenate()
    test_log_softmax()
    test_batchnorm()
    test_dense()
    test_relu()
    test_sym_scalar_pow()
    test_scalar_sym_pow()
    test_exp()
    test_log()
    test_tanh()
    test_sigmoid()
    test_softmax()
    test_squeeze()
    test_pad()
    test_matmul()

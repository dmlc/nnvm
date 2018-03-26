import numpy as np
import tvm
from tvm.contrib import graph_runtime
import topi.testing
import nnvm.symbol as sym
import nnvm.compiler
from nnvm.testing.config import ctx_list


def helper(symbol, inputs, dtype,
           np_forward, np_backward=None, need_input=True, need_head_grads=True):
    ishapes = {}
    input_syms = []
    np_inputs = {}
    for (name, shape, s) in inputs:
        ishapes.update({name: shape})
        np_inputs.update({name: np.random.uniform(size=shape).astype(dtype)})
        input_syms.append(s)

    for target, ctx in ctx_list():
        graph, lib, _ = nnvm.compiler.build(symbol, target, ishapes)
        m = graph_runtime.create(graph, lib, ctx)
        m.run(**np_inputs)
        y_np = np_forward(**np_inputs)
        out = m.get_output(0, tvm.nd.empty(y_np.shape, dtype))
        np.testing.assert_allclose(out.asnumpy(), y_np, atol=1e-5, rtol=1e-5)
        # backward
        if np_backward:
            graph._set_symbol_list_attr("grad_ys", symbol)
            graph._set_symbol_list_attr("grad_xs", input_syms)
            graph._set_symbol_list_attr("grad_ys_out_grad", sym.Variable("head_grads", shape=y_np.shape))
            graph = graph.apply("Gradient")
            ishapes.update({"head_grads": y_np.shape})
            graph, lib, _ = nnvm.compiler.build(graph, target, ishapes)
            m = graph_runtime.create(graph, lib, ctx)
            head_grads = np.random.uniform(size=y_np.shape).astype(dtype)
            y_np = np_backward(head_grads=head_grads, **np_inputs)
            b_inputs = {}
            if need_input:
                b_inputs.update(np_inputs)
            if need_head_grads:
                b_inputs.update({"head_grads":head_grads})
            m.run(**b_inputs)
            for i in range(len(y_np)):
                out = m.get_output(i, tvm.nd.empty(y_np[i].shape, dtype))
                np.testing.assert_allclose(out.asnumpy(), y_np[i], atol=1e-5, rtol=1e-5)


def test_relu():
    x = sym.Variable("x")
    y = sym.relu(sym.leaky_relu(x, alpha=0.3) - 0.2)

    def forward(x):
        x = (x < 0) * x * 0.3 + (x > 0) * x - 0.2
        return (x > 0) * x

    def backward(head_grads, x):
        sub = (x < 0) * x * 0.3 + (x > 0) * x - 0.2
        return [(sub > 0).astype("float") * \
                ((x > 0).astype("float") + 0.3 * (x < 0).astype("float")) * head_grads]

    dtype = "float32"
    dshape = (1, 3, 32, 32)
    inputs = [('x', dshape, x)]
    helper(y, inputs, dtype, forward, backward)


def test_sym_scalar_pow():
    scalar = 3
    x = sym.Variable("x")
    y = x**scalar

    def forward(x):
        return x**scalar

    def backward(head_grads, x):
        return [scalar * x**(scalar -  1) * head_grads]

    dtype = "float32"
    dshape = (1, 3, 32, 32)
    inputs = [('x', dshape, x)]
    helper(y, inputs, dtype, forward, backward)


def test_scalar_sym_pow():
    scalar = 3
    x = sym.Variable("x")
    y = scalar**x

    def forward(x):
        return scalar**x

    def backward(head_grads, x):
        return [np.log(scalar) * scalar**x * head_grads]

    dtype = "float32"
    dshape = (1, 3, 32, 32)
    inputs = [('x', dshape, x)]
    helper(y, inputs, dtype, forward, backward)


def test_exp():
    x = sym.Variable("x")
    y = sym.exp(x)

    def forward(x):
        return np.exp(x)

    def backward(head_grads, x):
        return [np.exp(x) * head_grads]

    dtype = "float32"
    dshape = (1, 3, 32, 32)
    inputs = [('x', dshape, x)]
    helper(y, inputs, dtype, forward, backward)


def test_log():
    x = sym.Variable("x")
    y = sym.log(x)

    def forward(x):
        return np.log(x)

    def backward(head_grads, x):
        return [1. / x * head_grads]

    dtype = "float32"
    dshape = (1, 3, 32, 32)
    inputs = [('x', dshape, x)]
    helper(y, inputs, dtype, forward, backward)


def test_tanh():
    x = sym.Variable("x")
    y = sym.tanh(x)

    def forward(x):
        return np.sinh(x) / np.cosh(x)

    def backward(head_grads, x):
        y_np = forward(x)
        return [(1 - y_np**2) * head_grads]

    dtype = "float32"
    dshape = (1, 3, 32, 32)
    inputs = [('x', dshape, x)]
    helper(y, inputs, dtype, forward, backward)


def test_sigmoid():
    x = sym.Variable("x")
    y = sym.sigmoid(x)

    def forward(x):
        return 1.0 / (1.0 + np.exp(-x))

    def backward(head_grads, x):
        y_np = forward(x)
        return [y_np *(1 - y_np) * head_grads]

    dtype = "float32"
    dshape = (1, 3, 32, 32)
    inputs = [('x', dshape, x)]
    helper(y, inputs, dtype, forward, backward)


def test_softmax():
    x = sym.Variable("x")
    y = sym.softmax(x)

    def forward(x):
        return topi.testing.softmax_python(x)

    def backward(head_grads, x):
        y = topi.testing.softmax_python(x)
        grad = y * (head_grads - np.sum(y * head_grads, axis=1, keepdims=True))
        return [grad]

    dtype = "float32"
    dshape = (10, 1000)
    inputs = [('x', dshape, x)]
    helper(y, inputs, dtype, forward), backward


def test_log_softmax():
    x = sym.Variable("x")
    y = sym.log_softmax(x)

    def forward(x):
        return topi.testing.log_softmax_python(x)

    def backward(head_grads, x):
        y = topi.testing.log_softmax_python(x)
        grad = head_grads - np.sum(y * head_grads, axis=1, keepdims=True)
        return [grad]

    dtype = "float32"
    dshape = (10, 1000)
    inputs = [('x', dshape, x)]
    helper(y, inputs, dtype, forward, backward)


def test_dense():
    x = sym.Variable("x", shape=(10, 100))
    w = sym.Variable("dense_weight", shape=(3, 100))
    b = sym.Variable("dense_bias", shape=(3,))
    y = sym.dense(x, w, b, use_bias=True, units=3, name="dense")
    y = sym.flatten(y)

    def forward(x, dense_weight, dense_bias):
        return np.dot(x, dense_weight.T) + dense_bias
    dtype = "float32"
    inputs = [
        ('x', (10, 100), x),
        ('dense_weight', (3, 100), w),
        ('dense_bias', (3,), b)
    ]
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
    inputs = [
        ('x', (10, 20), x),
        ('gamma', (20,), gamma),
        ('beta', (20,), beta),
        ('moving_mean', (20,), moving_var),
        ('moving_var', (20,), moving_mean)
    ]

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

    def backward(head_grads, x):
        return [np.reshape(head_grads, x.shape)]

    dtype = "float32"
    inputs = [('x', dshape, x)]
    helper(y, inputs, dtype, forward, backward)


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
    inputs = [('x', (1, 3, 28, 28), x)]
    helper(y, inputs, dtype, forward)


def test_clip():
    x = sym.Variable("x")
    a_min=0.2
    a_max=0.75
    y = sym.clip(x, a_min=a_min, a_max=a_max)

    def forward(x):
        return np.clip(x, a_min=a_min, a_max=a_max)

    def backward(head_grads, x):
        mask1 = np.greater_equal(x, a_min).astype("float")
        mask2 = np.less_equal(x, a_max).astype("float")
        return [head_grads * mask1 * mask2]


    dtype = "float32"
    inputs = [('x', (3, 4, 5), x)]
    helper(y, inputs, dtype, forward, backward)

def verify_elemwise_sum(num_args):
    s = [sym.Variable("input" + str(i)) for i in range(num_args)]
    y = sym.elemwise_sum(*s, num_args=num_args)

    def forward(**inputs):
        return np.sum(np.array(inputs.values()), axis=0)

    def backward(head_grads, **inputs):
        return [head_grads] * num_args

    dtype = "float32"
    inputs = [("input" + str(i), (3, 4, 5), s[i])
              for i in range(num_args)]
    helper(y, inputs, dtype, forward, backward, need_input=False)


def test_elemwise_sum():
    verify_elemwise_sum(1)
    verify_elemwise_sum(5)
    verify_elemwise_sum(7)


def test_full():
    shape = (3, 4, 5)
    value = 7
    dtype = "float32"
    for target, ctx in ctx_list():
        data = sym.Variable("data", dtype=dtype)
        # full_like
        s = sym.full_like(data=data, fill_value=value, name="s")
        graph, lib, _ = nnvm.compiler.build(s, target, {"data": shape})
        m = graph_runtime.create(graph, lib, ctx)
        m.run(data=np.random.uniform(size=shape).astype(dtype))
        out = m.get_output(0, tvm.nd.empty(shape, dtype=dtype))
        np.testing.assert_allclose(
            out.asnumpy(),
            np.full(shape, fill_value=value, dtype=dtype),
            atol=1e-5, rtol=1e-5)
        # ones_like
        s = sym.ones_like(data=data, fill_value=value, name="s")
        graph, lib, _ = nnvm.compiler.build(s, target, {"data": shape})
        m = graph_runtime.create(graph, lib, ctx)
        m.run(data=np.random.uniform(size=shape).astype(dtype))
        out = m.get_output(0, tvm.nd.empty(shape, dtype=dtype))
        np.testing.assert_allclose(
            out.asnumpy(),
            np.full(shape, fill_value=1, dtype=dtype),
            atol=1e-5, rtol=1e-5)
        # zeros_like
        s = sym.zeros_like(data=data, fill_value=value, name="s")
        graph, lib, _ = nnvm.compiler.build(s, target, {"data": shape})
        m = graph_runtime.create(graph, lib, ctx)
        m.run(data=np.random.uniform(size=shape).astype(dtype))
        out = m.get_output(0, tvm.nd.empty(shape, dtype=dtype))
        np.testing.assert_allclose(
            out.asnumpy(),
            np.full(shape, fill_value=0, dtype=dtype),
            atol=1e-5, rtol=1e-5)
        # full
        s = sym.full(shape=shape, dtype=dtype, fill_value=value, name="s")
        graph, lib, _ = nnvm.compiler.build(s, target)
        m = graph_runtime.create(graph, lib, ctx)
        m.run()
        out = m.get_output(0, tvm.nd.empty(shape, dtype=dtype))
        np.testing.assert_allclose(
            out.asnumpy(),
            np.full(shape, fill_value=value, dtype=dtype),
            atol=1e-5, rtol=1e-5)
        # ones
        s = sym.ones(shape=shape, dtype=dtype, name="s")
        graph, lib, _ = nnvm.compiler.build(s, target)
        m = graph_runtime.create(graph, lib, ctx)
        m.run()
        out = m.get_output(0, tvm.nd.empty(shape, dtype=dtype))
        np.testing.assert_allclose(
            out.asnumpy(),
            np.full(shape, fill_value=1, dtype=dtype),
            atol=1e-5, rtol=1e-5)
        # zeros
        s = sym.zeros(shape=shape, dtype=dtype, name="s")
        graph, lib, _ = nnvm.compiler.build(s, target)
        m = graph_runtime.create(graph, lib, ctx)
        m.run()
        out = m.get_output(0, tvm.nd.empty(shape, dtype=dtype))
        np.testing.assert_allclose(
            out.asnumpy(),
            np.full(shape, fill_value=0, dtype=dtype),
            atol=1e-5, rtol=1e-5)


def test_block_grad():
    x = sym.Variable("x")
    y = sym.block_grad(x)

    def forward(x):
        return x

    def backward(head_grads, x):
        return [np.zeros_like(head_grads)]


    dtype = "float32"
    inputs = [('x', (3, 4, 5), x)]
    helper(y, inputs, dtype, forward, backward, need_head_grads=False)


def test_greater():
    l = sym.Variable("l")
    r = sym.Variable("r")
    y = sym.greater(l, r)

    def forward(l, r):
        return np.greater(l, r).astype("float32")

    def backward(head_grads, l, r):
        return [np.zeros_like(l)]


    dtype = "float32"
    inputs = [('l', (3, 4, 5), l),
              ('r', (3, 4, 5), r)]
    helper(y, inputs, dtype, forward, backward, need_head_grads=False)


def test_less():
    l = sym.Variable("l")
    r = sym.Variable("r")
    y = sym.less(l, r)

    def forward(l, r):
        return np.less(l, r).astype("float32")

    def backward(head_grads, l, r):
        return [np.zeros_like(l)]


    dtype = "float32"
    inputs = [('l', (3, 4, 5), l),
              ('r', (3, 4, 5), r)]
    helper(y, inputs, dtype, forward, backward, need_head_grads=False)


def test_reshape_like():
    x = sym.Variable("x")
    y = sym.Variable("y")
    z = sym.reshape_like(x, y)

    def forward(x, y):
        return np.reshape(x, y.shape)

    def backward(head_grads, x, y):
        return [np.reshape(head_grads, x.shape),
                np.zeros_like(y)]


    dtype = "float32"
    inputs = [('x', (3, 4, 5), x),
              ('y', (5, 4, 3), y)]
    helper(z, inputs, dtype, forward, backward)


def verify_expand_like(in_shape, out_shape, axis, exclude):
    x = sym.Variable("x")
    y = sym.Variable("y")
    z = sym.expand_like(x, y, axis=axis, exclude=exclude)

    def forward(x, y):
        odim = len(out_shape)
        real_axis = [i if i >= 0 else i + odim for i in axis]
        real_axis = sorted(real_axis)
        if exclude:
            real_axis = list(set(range(odim)) - set(real_axis))
        for i in real_axis:
            x = np.expand_dims(x, i).astype(x.dtype)
        for i in real_axis:
            x = np.concatenate([x]*out_shape[i], axis=i).astype(x.dtype)

        return x

    def backward(head_grads, x, y):
        odim = len(out_shape)
        real_axis = [i if i >= 0 else i + odim for i in axis]
        real_axis = sorted(real_axis)
        if exclude:
            real_axis = list(set(range(odim)) - set(real_axis))
        return [np.sum(head_grads, axis=tuple(real_axis)),
                np.zeros_like(y)]


    dtype = "float32"
    inputs = [('x', in_shape, x),
              ('y', out_shape, y)]
    helper(z, inputs, dtype, forward, backward, need_input=False)


def test_expand_like():
    verify_expand_like((3,), (3, 2), [1], False)
    verify_expand_like((2,), (2, 3), [1], False)
    verify_expand_like((3, 4), (3, 5, 4), [1], False)
    verify_expand_like((5, 7), (5, 6, 7, 8), [0, 2], True)


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
    test_clip()
    test_elemwise_sum()
    test_full()
    test_block_grad()
    test_greater()
    test_less()
    test_reshape_like()
    test_expand_like()

import nnvm.symbol as sym

def test_compose():
    x = sym.Variable('x')
    z = sym.Variable('z')
    y = sym.exp(sym.add(x, x, name='add', gpu=2),
                name='exp', gpu=1, attr={"kk": "1"})

    assert y.list_arguments() == ['x']
    assert y.list_outputs() == ["exp_output"]
    assert y.list_attr()['gpu'] == '1'
    z = y.get_internals()
    assert z['add_output'].list_outputs() == ['add_output']
    assert y.list_attr(recursive=True)['add_gpu'] == '2'


if __name__ == "__main__":
    test_compose()

import nnvm
import nnvm.compiler.optimizer as optimizer
import nnvm.compiler.lr_scheduler as lr_scheduler

def test_FactorScheduler():
    scheduler = lr_scheduler.FactorScheduler(step=5, factor=0.1)
    num_updates = nnvm.sym.Variable('update_t')
    update_t = nnvm.sym.assign(num_updates, num_updates + 1)
    update_lr = scheduler(update_t)
    g = nnvm.graph.create(update_lr).ir()
    assert '__rpow_scalar__' in g
    assert 'clip' in g

def test_sgd():
    x = nnvm.sym.Variable("x")
    y = nnvm.sym.Variable("y")
    z = nnvm.sym.elemwise_add(x, nnvm.sym.sqrt(y))

    scheduler = lr_scheduler.FactorScheduler(step=10, factor=0.5)
    opt = optimizer.SGD(learning_rate = 0.001, lr_scheduler=scheduler,
                        rescale_grad=0.2, clip_gradient=1.0, wd=0.1)
    opt_sym = opt.minimize(z)
    assert len(opt_sym.list_output_names()) == 2
    g = nnvm.graph.create(opt_sym).ir()
    assert 'SGD_t' in g

def test_adam():
    x = nnvm.sym.Variable('x')
    y = nnvm.sym.elemwise_add(x, x, name='add1')
    y = nnvm.sym.flatten(y, name="flatten1")

    scheduler = lr_scheduler.FactorScheduler(step=5, factor=0.6)
    opt = optimizer.Adam(lr_scheduler=scheduler,
                         rescale_grad=0.2, clip_gradient=1.0, wd=0.2)
    opt_sym = opt.minimize(y)
    assert len(opt_sym.list_output_names()) == 1
    g = nnvm.graph.create(opt_sym).ir()
    assert 'Adam_t' in g
    assert 'Adam_m0' in g
    assert 'Adam_v0' in g

if __name__ == "__main__":
    test_FactorScheduler()
    test_sgd()
    test_adam()

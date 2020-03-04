from chainer import optimizers


def set_optimizer(model, alpha=0.0002, beta1=0.5, beta2=0.999):
    optimizer = optimizers.Adam(alpha=alpha, beta1=beta1, beta2=beta2)
    optimizer.setup(model)

    return optimizer
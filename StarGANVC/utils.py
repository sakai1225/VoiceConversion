import chainer

from chainer import optimizers, cuda

xp = cuda.cupy


def set_optimizer(model, alpha, beta=0.5):
    optimizer = optimizers.Adam(alpha=alpha, beta1=beta)
    optimizer.setup(model)

    return optimizer


def call_zeros(tensor):
    zeros = xp.zeros_like(tensor).astype(xp.float32)
    zeros = chainer.as_variable(zeros)

    return zeros

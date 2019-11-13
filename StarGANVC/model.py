import chainer
import chainer.links as L
import chainer.functions as F
from chainer import cuda, Chain, initializers
import numpy as np

xp = cuda.cupy
cuda.get_device(0).use()


def glu(x):
    a, b = F.split_axis(x, 2, axis=1)

    return a * F.sigmoid(b)


class ConditionalInstanceNormalization(Chain):
    def __init__(self, in_ch, adv_type='sat'):
        w = initializers.GlorotUniform()
        if adv_type == 'sat':
            domain_ch = 8
        elif adv_type == 'orig':
            domain_ch = 4
        else:
            raise AttributeError

        super(ConditionalInstanceNormalization, self).__init__()
        with self.init_scope():
            self.gamma = L.Linear(domain_ch, in_ch, initialW=w)
            self.beta = L.Linear(domain_ch, in_ch, initialW=w)

    def __call__(self, x, code):
        mean = F.mean(x, axis=2, keepdims=True)
        var = F.mean((x - F.broadcast_to(mean, x.shape)) * (x - F.broadcast_to(mean, x.shape)), axis=2, keepdims=True)
        std = F.sqrt(var + 1e-8)

        width = x.shape[2]

        gamma = F.tile(F.expand_dims(self.gamma(code), axis=2), (1, 1, width))
        beta = F.tile(F.expand_dims(self.beta(code), axis=2), (1, 1, width))

        h = (x - mean) / std
        h = h * gamma + beta

        return h


class C2BG(Chain):
    def __init__(self, in_ch, out_ch, up=False, down=False):
        super(C2BG, self).__init__()
        w = initializers.GlorotUniform()
        self.up = up
        self.down = down
        with self.init_scope():
            self.cup = L.Convolution2D(in_ch, out_ch, 5, 1, 2, initialW=w)
            self.cpara = L.Convolution2D(in_ch, out_ch, 3, 1, 1, initialW=w)
            self.cdown = L.Convolution2D(in_ch, out_ch, 4, 2, 1, initialW=w)

            self.bn0 = L.BatchNormalization(out_ch)

    def __call__(self, x):
        if self.up:
            h = F.unpooling_2d(x, 2, 2, 0, cover_all=False)
            h = glu(self.bn0(self.cup(h)))

        elif self.down:
            h = glu(self.bn0(self.cdown(x)))

        else:
            h = glu(self.bn0(self.cpara(x)))

        return h


class C1BG(Chain):
    def __init__(self, in_ch, out_ch, adv_type='sat', up=False, down=False):
        super(C1BG, self).__init__()
        w = initializers.GlorotUniform()
        self.up = up
        self.down = down
        with self.init_scope():
            self.cup = L.Convolution1D(in_ch, out_ch, 3, 1, 1, initialW=w)
            self.cpara = L.Convolution1D(in_ch, out_ch, 3, 1, 1, initialW=w)
            self.cdown = L.Convolution1D(in_ch, out_ch, 4, 2, 1, initialW=w)

            #self.bn0 = L.BatchNormalization(out_ch)
            self.cin0 = ConditionalInstanceNormalization(out_ch, adv_type)

    def __call__(self, x, domain=None):
        if self.up:
            h = F.unpooling_1d(x, 2, 2, 0, cover_all=False)
            h = glu(self.bn0(self.cup(h)))

        if self.down:
            h = glu(self.bn0(self.cdown(x)))

        else:
            h = glu(self.cin0(self.cpara(x), domain))
            #h = glu(self.bn0(self.cpara(x)))

        return h


class ResBlock(Chain):
    def __init__(self, in_ch, out_ch, adv_type='sat'):
        w = initializers.GlorotUniform()
        super(ResBlock, self).__init__()
        with self.init_scope():
            self.cbg0 = C1BG(in_ch, out_ch, adv_type)
            self.c0 = L.Convolution1D(in_ch, in_ch, 3, 1, 1, initialW=w)
            #self.bn0 = L.BatchNormalization(in_ch)
            self.bn0 = ConditionalInstanceNormalization(in_ch, adv_type)

    def __call__(self, x, domain):
        h = self.cbg0(x, domain)
        #h = self.cbg0(x)
        h = self.bn0(self.c0(h), domain)
        #h = self.bn0(self.c0(h))

        return h


class Generator(Chain):
    def __init__(self, base=128):
        w = initializers.GlorotUniform()
        super(Generator, self).__init__()
        with self.init_scope():
            self.c0 = L.Convolution2D(1 + 4, base, (15, 5), 1, (7, 2), initialW=w)
            self.cbg0 = C2BG(int(base/2), base*2, down=True)
            self.cbg1 = C2BG(base, base*4, down=True)

            self.c1 = L.Convolution1D(2304, base*2, 1, 1, 0, initialW=w)
            self.bn1 = L.BatchNormalization(base*2)

            self.res0 = ResBlock(base*2, base*4)
            self.res1 = ResBlock(base*2, base*4)
            self.res2 = ResBlock(base*2, base*4)
            self.res3 = ResBlock(base*2, base*4)
            self.res4 = ResBlock(base*2, base*4)
            self.res5 = ResBlock(base*2, base*4)
            self.res6 = ResBlock(base*2, base*4)
            self.res7 = ResBlock(base*2, base*4)
            self.res8 = ResBlock(base*2, base*4)

            self.c2 = L.Convolution1D(base*2, 2304, 1, 1, 0, initialW=w)
            self.bn2 = L.BatchNormalization(2304)

            self.cbg2 = C2BG(base*2 + 4, base*8, up=True)
            self.cbg3 = C2BG(base*4 + 4, 72, up=True)

            self.c3 = L.Convolution2D(36, 1, 3, 1, 1, initialW=w)

    def _prepare(self, feat, domain):
        batch, ch, height, width = feat.shape
        _, label = domain.shape
        domain_map = F.broadcast_to(domain, (height, width, batch, label))
        domain_map = F.transpose(domain_map, (2, 3, 0, 1))

        return F.concat([feat, domain_map], axis=1)

    def __call__(self, x, domain):
        batch, ch, height, width = x.shape
        _, label = domain.shape
        domain_map = F.broadcast_to(domain, (height, width, batch, label))
        domain_map = F.transpose(domain_map, (2, 3, 0, 1))
        x = F.concat([x, domain_map], axis=1)

        b = x.shape[0]
        h = glu(self.c0(x))
        h = self.cbg0(h)
        h = self.cbg1(h)
        h = F.transpose(h, (0, 1, 3, 2)).reshape(b, 2304, 32)
        h = self.bn1(self.c1(h))
        h = self.res0(h)
        h = self.res1(h)
        h = self.res2(h)
        h = self.res3(h)
        h = self.res4(h)
        h = self.res5(h)
        h = self.res6(h)
        h = self.res7(h)
        h = self.res8(h)
        h = self.bn2(self.c2(h))
        h = F.transpose(F.reshape(h, (b, 256, 9, 32)), (0, 1, 3, 2))
        h = self.cbg2(self._prepare(h, domain))
        h = self.cbg3(self._prepare(h, domain))
        h = self.c3(h)

        return h


class GeneratorWithCIN(Chain):
    def __init__(self, base=128, adv_type='sat'):
        w = initializers.GlorotUniform()
        super(GeneratorWithCIN, self).__init__()
        with self.init_scope():
            self.c0 = L.Convolution2D(1, base, (15, 5), 1, (7, 2), initialW=w)
            self.cbg0 = C2BG(int(base/2), base*2, down=True)
            self.cbg1 = C2BG(base, base*4, down=True)

            self.c1 = L.Convolution1D(2304, base*2, 1, 1, 0, initialW=w)
            self.bn1 = L.BatchNormalization(base*2)

            self.res0 = ResBlock(base*2, base*4, adv_type)
            self.res1 = ResBlock(base*2, base*4, adv_type)
            self.res2 = ResBlock(base*2, base*4, adv_type)
            self.res3 = ResBlock(base*2, base*4, adv_type)
            self.res4 = ResBlock(base*2, base*4, adv_type)
            self.res5 = ResBlock(base*2, base*4, adv_type)
            self.res6 = ResBlock(base*2, base*4, adv_type)
            self.res7 = ResBlock(base*2, base*4, adv_type)
            self.res8 = ResBlock(base*2, base*4, adv_type)

            self.c2 = L.Convolution1D(base*2, 2304, 1, 1, 0, initialW=w)
            self.bn2 = L.BatchNormalization(2304)

            self.cbg2 = C2BG(base*2, base*8, up=True)
            self.cbg3 = C2BG(base*4, 72, up=True)

            self.c3 = L.Convolution2D(36, 1, 3, 1, 1, initialW=w)

    def _prepare(self, feat, domain):
        batch, ch, height, width = feat.shape
        _, label = domain.shape
        domain_map = F.broadcast_to(domain, (height, width, batch, label))
        domain_map = F.transpose(domain_map, (2, 3, 0, 1))

        return F.concat([feat, domain_map], axis=1)

    def __call__(self, x, domain):
        b = x.shape[0]
        h = glu(self.c0(x))
        h = self.cbg0(h)
        h = self.cbg1(h)
        h = F.transpose(h, (0, 1, 3, 2)).reshape(b, 2304, 32)
        h = self.bn1(self.c1(h))
        h = self.res0(h, domain)
        h = self.res1(h, domain)
        h = self.res2(h, domain)
        h = self.res3(h, domain)
        h = self.res4(h, domain)
        h = self.res5(h, domain)
        h = self.res6(h, domain)
        h = self.res7(h, domain)
        h = self.res8(h, domain)
        h = self.bn2(self.c2(h))
        h = F.transpose(F.reshape(h, (b, 256, 9, 32)), (0, 1, 3, 2))
        h = self.cbg2(h)
        h = self.cbg3(h)
        h = self.c3(h)

        return h


class BG(Chain):
    def __init__(self, out_ch):
        super(BG, self).__init__()
        with self.init_scope():
            self.bn0 = L.BatchNormalization(out_ch)

    def __call__(self, x):
        h = glu(self.bn0(x))

        return h


class Discriminator(Chain):
    def __init__(self, base=64):
        w = initializers.GlorotUniform()
        super(Discriminator, self).__init__()

        with self.init_scope():
            self.c0 = L.Convolution2D(1, base*2, 3, 1, 1, initialW=w)
            self.c1 = L.Convolution2D(base, base*4, 3, 2, 1, initialW=w)
            self.bg1 = BG(base*4)
            self.c2 = L.Convolution2D(base*2, base*8, 3, 2, 1, initialW=w)
            self.bg2 = BG(base*8)
            self.c3 = L.Convolution2D(base*4, base*16, 3, 2, 1, initialW=w)
            self.bg3 = BG(base*16)
            self.c4 = L.Convolution2D(base*8, base*16, (5, 1), 1, (2, 0), initialW=w)
            self.bg4 = BG(base*16)
            self.c5 = L.Convolution2D(base*8, 1, (3, 1), 1, (1, 0), initialW=w)

            self.lembed = L.Linear(None, base*8, nobias=True, initialW=w)
            self.l1 = L.Linear(None, 1, initialW=w)

    def __call__(self, x, category):
        h = glu(self.c0(x))
        h = self.bg1(self.c1(h))
        h = self.bg2(self.c2(h))
        h = self.bg3(self.c3(h))
        h = self.bg4(self.c4(h))
        h = F.sum(h, axis=(2, 3))
        hout = self.l1(h)
        cat = self.lembed(category)
        hout += F.sum(cat * h, axis=1, keepdims=True)

        return hout

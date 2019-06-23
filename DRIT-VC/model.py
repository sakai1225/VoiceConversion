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

class MLP(Chain):
    def __init__(self, base=256):
        super(MLP, self).__init__()

        with self.init_scope():
            self.l0 = L.Linear(8, base)
            self.l1 = L.Linear(base, base)
            self.l2 = L.Linear(base, base*4)

    def __call__(self, x):
        h = F.relu(self.l0(x))
        h = F.relu(self.l1(h))
        h = self.l2(h)

        return h

class C2BG(Chain):
    """2D convolution -> Batch Normalization -> Gated linear unit

    """
    def __init__(self, in_ch, out_ch, up=False, down=False):
        super(C2BG, self).__init__()
        w = initializers.Normal(0.02)
        self.up = up
        self.down = down
        with self.init_scope():
            self.cup = L.Convolution2D(in_ch, out_ch,5,1,2,initialW=w)
            self.cpara = L.Convolution2D(in_ch, out_ch,3,1,1,initialW=w)
            self.cdown = L.Convolution2D(in_ch, out_ch, 4,2,1,initialW=w)

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
    """1D convolution -> Batch Normalization -> Gated linear unit
    
    """
    def __init__(self, in_ch, out_ch, up=False, down=False):
        super(C1BG, self).__init__()
        w = initializers.Normal(0.02)
        self.up = up
        self.down = down
        with self.init_scope():
            self.cup = L.Convolution1D(in_ch, out_ch,3,1,1,initialW=w)
            self.cpara = L.Convolution1D(in_ch, out_ch,3,1,1,initialW=w)
            self.cdown = L.Convolution1D(in_ch, out_ch, 4,2,1,initialW=w)

            self.bn0 = L.BatchNormalization(out_ch)

    def __call__(self, x):
        if self.up:
            h = F.unpooling_1d(x, 2, 2, 0, cover_all=False)
            h = glu(self.bn0(self.cup(h)))

        if self.down:
            h = glu(self.bn0(self.cdown(x)))

        else:
            h = glu(self.bn0(self.cpara(x)))

        return h

class ResBlock(Chain):
    def __init__(self, in_ch, out_ch):
        w = initializers.Normal(0.02)
        super(ResBlock, self).__init__()
        with self.init_scope():
            self.cbg0 = C1BG(in_ch, out_ch)
            self.c0 = L.Convolution1D(in_ch, in_ch, 3,1,1,initialW=w)
            self.bn0 = L.BatchNormalization(in_ch)

    def __call__(self, x):
        h = self.cbg0(x)
        h = self.bn0(self.c0(h))

        return h + x

class ConcatResBlock(Chain):
    def __init__(self, base=256, attn_dim=256):
        super(ConcatResBlock, self).__init__()

        with self.init_scope():
            self.c0 = L.Convolution2D(base, base, 3, 1, 1)
            self.bn0 = L.BatchNormalization(base)
            self.c1 = L.Convolution2D(base, base, 3, 1, 1)
            self.bn1 = L.BatchNormalization(base)

            self.attr0 = L.Convolution2D(base + attn_dim, base + attn_dim, 1, 1, 0)
            self.attr1 = L.Convolution2D(base + attn_dim, base, 1, 1, 0)

            self.attr2 = L.Convolution2D(base + attn_dim, base + attn_dim, 1, 1, 0)
            self.attr3 = L.Convolution2D(base + attn_dim, base, 1, 1, 0)

    def __call__(self, x, z):
        batch, channel, height, width = z.shape[0], z.shape[1], x.shape[2], x.shape[3]
        z = F.tile(z.reshape(batch, channel, 1, 1), (1, 1, height, width))
        h = F.relu(self.bn0(self.c0(x)))
        h = F.relu(self.attr0(F.concat([h, z], axis=1)))
        h = F.relu(self.attr1(h))
        h = F.relu(self.bn1(self.c1(h)))
        h = F.relu(self.attr2(F.concat([h, z], axis=1)))
        h = F.relu(self.attr3(h))

        return h + x

class ContentEncoder(Chain):
    def __init__(self, base=128):
        w = initializers.Normal(0.02)
        super(ContentEncoder, self).__init__()
        with self.init_scope():
            self.c0 = L.Convolution2D(1, base, (15, 5), 1, (7, 2), initialW=w)
            self.cbg0 = C2BG(int(base/2), base*2, down=True)
            self.cbg1 = C2BG(base, base*4, down=True)
            
            self.c1 = L.Convolution1D(2304, base*2, 1, 1, 0, initialW=w)
            self.bn1 = L.BatchNormalization(base*2)

            self.res0 = ResBlock(base*2, base*4)
            self.res1 = ResBlock(base*2, base*4)
            self.res2 = ResBlock(base*2, base*4)
            self.res3 = ResBlock(base*2, base*4)
            #self.res4 = ResBlock(base*2, base*4)
            #self.res5 = ResBlock(base*2, base*4)

            self.c2 = L.Convolution1D(base*2, 2304, 1, 1, 0, initialW=w)
            self.bn2 = L.BatchNormalization(2304)

    def __call__(self, x):
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
        #h = self.res4(h)
        #h = self.res5(h)
        h = self.bn2(self.c2(h))
        h = F.transpose(F.reshape(h, (b, 256, 9, 32)), (0, 1, 3, 2))

        return h

class AttributeEncoder(Chain):
    def __init__(self, base=64):
        super(AttributeEncoder, self).__init__()
        w = initializers.Normal(0.02)
        with self.init_scope():
            self.c0 = L.Convolution2D(1, base, (15, 5), 1, (7, 2), initialW=w)
            self.cbg0 = C2BG(int(base/2), base*2, down=True)
            self.cbg1 = C2BG(base, base*4, down=True)
            self.cbg2 = C2BG(base*2, base*4, down=True)
            self.cbg3 = C2BG(base*2, base*4, down=True)
            self.c4 = L.Convolution2D(base*2, 8, 1, 1, 0)

    def __call__(self, x):
        h = glu(self.c0(x))
        h = self.cbg0(h)
        h = self.cbg1(h)
        h = self.cbg2(h)
        h = self.cbg3(h)
        batch, _, height, width = h.shape
        h = F.average_pooling_2d(h, (height, width))
        h = self.c4(h)
        h = F.reshape(h, (batch, 8))

        return h

class Decoder(Chain):
    def __init__(self, base=64):
        super(Decoder, self).__init__()
        w = initializers.Normal(0.02)
        with self.init_scope():
            self.dec0 = ConcatResBlock()
            self.dec1 = ConcatResBlock()
            self.dec2 = ConcatResBlock()
            self.dec3 = ConcatResBlock()

            self.up0 = C2BG(base*4, base*8, up=True)
            self.up1 = C2BG(base*4, 72, up=True)
            self.c0 = L.Convolution2D(36, 1, 3, 1, 1, initialW=w)

            self.mlp = MLP()

    def __call__(self, x, z):
        z = self.mlp(z)
        z0, z1, z2, z3 = F.split_axis(z, 4, axis=1)
        h = self.dec0(x, z0)
        h = self.dec1(h, z1)
        h = self.dec2(h, z2)
        h = self.dec3(h, z3)
        h = self.up0(h)
        h = self.up1(h)
        h = self.c0(h)

        return h

class Generator(Chain):
    def __init__(self):
        super(Generator, self).__init__()

        with self.init_scope():
            self.content_enc_x = ContentEncoder()
            self.content_enc_y = ContentEncoder()
            self.attr_enc_x = AttributeEncoder()
            self.attr_enc_y = AttributeEncoder()
            self.dec_x = Decoder()
            self.dec_y = Decoder()

    def _calc_mean_var(self, x):
        m = F.mean(x, axis=1, keepdims=True)
        v = F.mean((x - F.broadcast_to(m, x.shape))*(x - F.broadcast_to(m, x.shape)), axis = 1)

        return m, F.log(v)

    def _reconstruct(self, content, attr, switch='x'):
        mean, ln_var = self._calc_mean_var(attr)
        if switch == 'x':
            return self.dec_x(content, attr), mean, ln_var

        else:
            return self.dec_y(content, attr), mean, ln_var

    def _get_random(self, enc, nz=8):
        batch = enc.shape[0]
        z = chainer.as_variable(xp.random.normal(size=(batch, nz)).astype(xp.float32))

        return z

    def mock_inference(self, content, switch='x'):
        if switch == 'x':
            latent = self._get_random(content)
            y = self.dec_y(content, latent)
            y_attr = self.attr_enc_x(y)

            return latent, y, y_attr

        else:
            latent = self._get_random(content)
            x = self.dec_x(content, latent)
            x_attr = self.attr_enc_x(x)

            return latent, x, x_attr

    def inference(self, a):
        a_content = self.content_enc_x(a)
        latent = self._get_random(a_content)
        y = self.dec_y(a_content, latent)

        return y

    def __call__(self, a, b):
        ha_content = self.content_enc_x(a)
        ha_attribute = self.attr_enc_x(a)
        hb_content = self.content_enc_y(b)
        hb_attribute = self.attr_enc_y(b)

        ya = self.dec_x(hb_content, ha_attribute)
        yb = self.dec_y(ha_content, hb_attribute)

        ya_recon = self._reconstruct(ha_content, ha_attribute, switch='x')
        yb_recon = self._reconstruct(hb_content, hb_attribute, switch='y')

        ya_infer = self.mock_inference(ha_content, switch='y')
        yb_infer = self.mock_inference(hb_content, switch='x')

        a_out = (ha_content, ha_attribute, ya, ya_recon, ya_infer)
        b_out = (hb_content, hb_attribute, yb, yb_recon, yb_infer)

        return a_out, b_out

class BG(Chain):
    def __init__(self, out_ch):
        super(BG, self).__init__()
        with self.init_scope():
            self.bn0 = L.BatchNormalization(out_ch)

    def __call__(self, x):
        h = glu(self.bn0(x))

        return h

class DomainDiscriminator(Chain):
    def __init__(self,base=64):
        w = initializers.Normal(0.02)
        super(DomainDiscriminator, self).__init__()

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

    def __call__(self, x):
        h = glu(self.c0(x))
        h = self.bg1(self.c1(h))
        h = self.bg2(self.c2(h))
        h = self.bg3(self.c3(h))
        h = self.bg4(self.c4(h))
        h = self.c5(h)

        return h

class ContentDiscriminator(Chain):
    def __init__(self, base=256):
        super(ContentDiscriminator, self).__init__()
        w = initializers.Normal(0.02)

        with self.init_scope():
            self.c0 = C2BG(base, base, down=True)
            self.c1 = C2BG(int(base/2), base, down=True)
            self.c2 = C2BG(int(base/2), base)
            self.c3 = C2BG(int(base/2), base)
            self.c4 = L.Convolution2D(int(base/2), 1, 1, 1, 0)

    def __call__(self, x):
        h = self.c0(x)
        h = self.c1(h)
        h = self.c2(h)
        h = self.c3(h)
        h = self.c4(h)

        return h

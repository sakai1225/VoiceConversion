import torch
import torch.nn as nn

from torch.nn import init


def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('Linear') != -1:
        init.normal(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm2d') != -1:
        init.normal_(m.weight.data, 1.0, 0.02)
        init.constant_(m.bias.data, 0.0)


def weights_init_xavier(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        init.xavier_normal_(m.weight.data, gain=0.02)
    elif classname.find('Linear') != -1:
        init.xavier_normal_(m.weight.data, gain=0.02)
    elif classname.find('BatchNorm2d') != -1:
        init.normal_(m.weight.data, 1.0, 0.02)
        init.constant_(m.bias.data, 0.0)


def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
    elif classname.find('Linear') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
    elif classname.find('BatchNorm2d') != -1:
        init.normal_(m.weight.data, 1.0, 0.02)
        init.constant_(m.bias.data, 0.0)


def init_weights(net, init_type='normal'):
    print('initialization method [%s]' % init_type)
    if init_type == 'normal':
        net.apply(weights_init_normal)
    elif init_type == 'xavier':
        net.apply(weights_init_xavier)
    elif init_type == 'kaiming':
        net.apply(weights_init_kaiming)
    else:
        raise NotImplementedError('initialization method [%s] is not implemented' % init_type)


def calc_mean_std(feat, eps=1e-5):
    size = feat.size()
    N, C = size[:2]
    feat_var = feat.view(N, C, -1).var(dim=2) + eps
    feat_std = feat_var.sqrt().view(N, C, 1)
    feat_mean = feat.view(N, C, -1).mean(dim=2).view(N, C, 1)

    return feat_mean, feat_std


def adain(content_feat, style_feat):
    size = content_feat.size()
    style_mean, style_std = style_feat[:, :128], style_feat[:, 128:]
    style_mean = style_mean.unsqueeze(dim=2)
    style_std = style_std.unsqueeze(dim=2)
    content_mean, content_std = calc_mean_std(content_feat)

    normalized_feat = (content_feat - content_mean.expand(
        size)) / content_std.expand(size)

    return normalized_feat * style_std.expand(size) + style_mean.expand(size)


class Block(nn.Module):
    def __init__(self, in_ch, out_ch, kernel=3, stride=1, norm=False):
        super(Block, self).__init__()

        self.norm = norm

        if norm:
            self.block = nn.Sequential(
                nn.Conv1d(in_ch, out_ch, kernel, stride, 1),
                nn.ReLU(),
                nn.Dropout(p=0.0),
                nn.InstanceNorm1d(out_ch)
            )

        else:
            self.block = nn.Sequential(
                nn.Conv1d(in_ch, out_ch, kernel, stride, 1),
                nn.ReLU(),
                nn.Dropout(p=0.0)
            )

    def forward(self, x):
        return self.block(x)


class AdaINBlock(nn.Module):
    def __init__(self, in_ch, out_ch, up=False):
        super(AdaINBlock, self).__init__()
        self.up = up

        if up:
            self.block = nn.Sequential(
                nn.Upsample(scale_factor=2),
                nn.Conv1d(in_ch, out_ch, 3, 1, 1),
                nn.ReLU(),
                nn.Dropout(p=0.0)
            )

        else:
            self.block = nn.Sequential(
                nn.Conv1d(in_ch, out_ch, 3, 1, 1),
                nn.ReLU(),
                nn.Dropout(p=0.0)
            )

    def forward(self, x, z):
        h = self.block(x)
        h = adain(h, z)

        return h


class Connection(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(Connection, self).__init__()

        self.connection = nn.Sequential(
            nn.Linear(in_ch, out_ch),
            nn.ReLU(),
            nn.Linear(out_ch, out_ch),
            nn.ReLU()
        )

    def forward(self, x):
        return self.connection(x) + x


class AdaINConnection(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(AdaINConnection, self).__init__()

        self.block = nn.Sequential(
            Connection(in_ch, out_ch),
            Connection(out_ch, out_ch),
            Connection(out_ch, out_ch),
            Connection(out_ch, out_ch)
        )

    def forward(self, x):
        return self.block(x)


class AffineTransform(nn.Module):
    def __init__(self, in_ch, out_ch, up=False):
        super(AffineTransform, self).__init__()
        self.up = up
        self.c0 = nn.Linear(in_ch, out_ch)

    def forward(self, z):
        return self.c0(z)


class SpeakerEncoderBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(SpeakerEncoderBlock, self).__init__()

        self.c0 = nn.Sequential(
            Block(in_ch, out_ch),
            Block(out_ch, out_ch)
        )

        self.c1 = nn.Sequential(
            Block(out_ch, out_ch),
            Block(out_ch, out_ch, 4, 2)
        )

        self.pool = nn.AvgPool1d(2, 2, 0)

    def forward(self, x):
        h = self.c0(x)
        h = h + x
        h = self.c1(h)
        h = h + self.pool(x)

        return h


class ContentEncoderBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(ContentEncoderBlock, self).__init__()

        self.c0 = nn.Sequential(
            Block(in_ch, out_ch, norm=True),
            Block(out_ch, out_ch, norm=True)
        )

        self.c1 = nn.Sequential(
            Block(out_ch, out_ch, norm=True),
            Block(out_ch, out_ch, 4, 2, True)
        )

        self.pool = nn.AvgPool1d(2, 2, 0)

    def forward(self, x):
        h = self.c0(x)
        h = h + x
        h = self.c1(h)
        h = h + self.pool(x)

        return h


class DecoderBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(DecoderBlock, self).__init__()

        self.ab0 = AdaINBlock(in_ch, out_ch)
        self.ab1 = AdaINBlock(out_ch, out_ch)
        self.ab2 = AdaINBlock(out_ch, out_ch)
        self.ab3 = AdaINBlock(out_ch, out_ch, up=True)

        self.at0 = AffineTransform(in_ch, out_ch*2)
        self.at1 = AffineTransform(out_ch, out_ch*2)
        self.at2 = AffineTransform(out_ch, out_ch*2)
        self.at3 = AffineTransform(out_ch, out_ch*2, up=True)

        self.up = nn.Upsample(scale_factor=2)

    def forward(self, x, z):
        h = self.ab0(x, self.at0(z))
        h = self.ab1(h, self.at1(z))
        h = h + x
        h = self.ab2(h, self.at2(z))
        h = self.ab3(h, self.at3(z))
        h = h + self.up(x)

        return h


class SpeakerEncoder(nn.Module):
    def __init__(self, dim, base=128):
        super(SpeakerEncoder, self).__init__()

        self.convbank = nn.Sequential(
            nn.Conv1d(dim, base, 7, 1, 3),
            nn.ReLU()
        )

        self.blocks = nn.Sequential(
            SpeakerEncoderBlock(base, base),
            SpeakerEncoderBlock(base, base),
            SpeakerEncoderBlock(base, base)
        )

        self.conn = nn.Sequential(
            Connection(base, base),
            Connection(base, base)
        )

        self.gap = nn.AdaptiveAvgPool1d(1)

    def forward(self, x):
        h = self.convbank(x)
        h = self.blocks(h)
        h = self.gap(h)
        h = self.conn(h.squeeze(2))

        return h


class ContentEncoder(nn.Module):
    def __init__(self, dim, base=128):
        super(ContentEncoder, self).__init__()

        self.convbank = nn.Sequential(
            nn.Conv1d(dim, base, 7, 1, 3),
            nn.ReLU()
        )

        self.blocks = nn.Sequential(
            ContentEncoderBlock(base, base),
            ContentEncoderBlock(base, base),
            ContentEncoderBlock(base, base)
        )

        self.lm = nn.Conv1d(base, base, 1, 1, 0)
        self.ls = nn.Conv1d(base, base, 1, 1, 0)

    def forward(self, x):
        h = self.convbank(x)
        h = self.blocks(h)

        mu = self.lm(h)
        sigma = self.ls(h)

        return mu, sigma


class Decoder(nn.Module):
    def __init__(self, base=128):
        super(Decoder, self).__init__()

        self.conn = AdaINConnection(base, base)

        self.dec0 = DecoderBlock(base, base)
        self.dec1 = DecoderBlock(base, base)
        self.dec2 = DecoderBlock(base, base)
        self.out = nn.Conv1d(base, base*4, 1, 1, 0)

    def forward(self, x, z):
        z = self.conn(z)

        h = self.dec0(x, z)
        h = self.dec1(h, z)
        h = self.dec2(h, z)
        h = self.out(h)

        return h


class Model(nn.Module):
    def __init__(self, dim):
        super(Model, self).__init__()

        self.se = SpeakerEncoder(dim)
        self.ce = ContentEncoder(dim)
        self.dec = Decoder()

        init_weights(self.se)
        init_weights(self.ce)
        init_weights(self.dec)

    def forward(self, x, z):
        z = self.se(z)
        mu, sigma = self.ce(x)
        eps = sigma.new(*sigma.size()).normal_(0, 1)
        h = mu + torch.exp(sigma / 2) * eps

        h = self.dec(h, z)

        return h, mu, sigma

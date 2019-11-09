import torch
import torch.nn as nn

from torch.nn import init


def glu(x):
    h, gate = torch.chunk(x, 2, dim=1)
    h = h * torch.sigmoid(gate)

    return h


class ConditionalInstanceNorm1d(nn.Module):
    def __init__(self, ch, cls_num):
        super(ConditionalInstanceNorm1d, self).__init__()

        self.norm = nn.InstanceNorm1d(ch, affine=False)
        self.scale = nn.Linear(cls_num * 2, ch)
        self.bias = nn.Linear(cls_num*2, ch)

        init.xavier_uniform_(self.scale.weight.data)
        init.xavier_uniform_(self.bias.weight.data)

    def forward(self, x, c):
        h = self.norm(x)
        s = self.scale(c).unsqueeze(2).repeat(1, 1, h.size(2))
        b = self.bias(c).unsqueeze(2).repeat(1, 1, h.size(2))

        return h * s + b


class Down2d(nn.Module):
    def __init__(self, in_ch, out_ch, kernel, stride, padding):
        super(Down2d, self).__init__()

        self.c = nn.Conv2d(in_ch, out_ch, kernel_size=kernel, stride=stride, padding=padding)
        self.norm = nn.BatchNorm2d(out_ch)

        init.normal_(self.c.weight.data, 0.0, 0.02)
        init.normal_(self.norm.weight.data, 1.0, 0.02)
        init.constant_(self.norm.bias.data, 0)

    def forward(self, x):
        h = glu(self.norm(self.c(x)))

        return h


class Down1d(nn.Module):
    def __init__(self, cls_num, in_ch, out_ch, kernel, stride, padding):
        super(Down1d, self).__init__()

        self.c = nn.Conv1d(in_ch, out_ch, kernel_size=kernel, stride=stride, padding=padding)
        self.norm = ConditionalInstanceNorm1d(out_ch, cls_num)

        init.normal_(self.c.weight.data, 0.0, 0.02)

    def forward(self, x, c):
        h = self.c(x)
        h = self.norm(h, c)
        h = glu(h)

        return h


class Up2d(nn.Module):
    def __init__(self, in_ch, out_ch, kernel, stride, padding):
        super(Up2d, self).__init__()

        self.c = nn.Conv2d(in_ch, out_ch, kernel_size=kernel, stride=stride, padding=padding)
        self.ps = nn.PixelShuffle(2)

        init.normal_(self.c.weight.data, 0.0, 0.02)

    def forward(self, x):
        h = glu(self.ps(self.c(x)))

        return h


class Generator(nn.Module):
    def __init__(self, cls_num=4, base=64):
        super(Generator, self).__init__()

        self.cin = nn.Conv2d(1, base*2, kernel_size=(15, 5), stride=(1, 1), padding=(7, 2))
        self.down = nn.Sequential(
            Down2d(base, base*4, 4, 2, 1),
            Down2d(base*2, base*8, 4, 2, 1)
        )

        self.c1 = nn.Conv1d(2304, base*4, 1, 1, 0)
        self.norm1 = nn.BatchNorm1d(base*4)

        self.down1 = Down1d(cls_num, base*4, base*8, 5, 1, 2)
        self.down2 = Down1d(cls_num, base*4, base*8, 5, 1, 2)
        self.down3 = Down1d(cls_num, base*4, base*8, 5, 1, 2)
        self.down4 = Down1d(cls_num, base*4, base*8, 5, 1, 2)
        self.down5 = Down1d(cls_num, base*4, base*8, 5, 1, 2)
        self.down6 = Down1d(cls_num, base*4, base*8, 5, 1, 2)
        self.down7 = Down1d(cls_num, base*4, base*8, 5, 1, 2)
        self.down8 = Down1d(cls_num, base*4, base*8, 5, 1, 2)
        self.down9 = Down1d(cls_num, base*4, base*8, 5, 1, 2)

        self.c2 = nn.Conv1d(base*4, 2304, 1, 1, 0)

        self.up = nn.Sequential(
            Up2d(base*4, base*16, 5, 1, 2),
            Up2d(base*2, base*8, 5, 1, 2)
        )

        self.cout = nn.Conv2d(base, 1, kernel_size=(15, 5), stride=(1, 1), padding=(7, 2))

        init.normal_(self.cin.weight.data, 0.0, 0.02)
        init.normal_(self.c1.weight.data, 0.0, 0.02)
        init.normal_(self.c2.weight.data, 0.0, 0.02)
        init.normal_(self.cout.weight.data, 0.0, 0.02)
        init.normal_(self.norm1.weight.data, 1.0, 0.02)
        init.constant_(self.norm1.bias.data, 0)

    def forward(self, x, c):
        h = glu(self.cin(x))
        h = self.down(h)
        h = h.permute(0, 1, 3, 2).contiguous().view(h.size(0), 2304, 32)
        h = self.norm1(self.c1(h))
        h = self.down1(h, c)
        h = self.down2(h, c)
        h = self.down3(h, c)
        h = self.down4(h, c)
        h = self.down5(h, c)
        h = self.down6(h, c)
        h = self.down7(h, c)
        h = self.down8(h, c)
        h = self.down9(h, c)
        h = self.c2(h)
        h = h.contiguous().view(h.size(0), 256, 9, 32).permute(0, 1, 3, 2)
        h = self.up(h)
        h = self.cout(h)

        return h


class Discriminator(nn.Module):
    def __init__(self, cls_num, base=128):
        super(Discriminator, self).__init__()
        self.cin = nn.Conv2d(1, base*2, 3, 1, 1)

        self.down = nn.Sequential(
            Down2d(base, base*4, 3, 2, 1),
            Down2d(base*2, base*8, 3, 2, 1),
            Down2d(base*4, base*16, 3, 2, 1),
            Down2d(base*8, base*16, kernel=(5, 1), stride=(1, 1), padding=(2, 0))
        )

        self.embed = nn.Linear(cls_num*2, base*8)
        self.fout = nn.Linear(base*8, 1)

        init.xavier_uniform_(self.embed.weight.data)
        init.xavier_uniform_(self.fout.weight.data)
        init.normal_(self.cin.weight.data, 0.0, 0.02)

    def forward(self, x, c):
        h = glu(self.cin(x))
        h = self.down(h)
        h = torch.sum(h, dim=(2, 3))
        out = self.fout(h)
        embeded = self.embed(c)
        out += torch.sum(embeded*h, dim=1, keepdim=True)

        return out

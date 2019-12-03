import chainer
import chainer.links as L
import chainer.functions as F
import numpy as np
import argparse

from chainer import cuda, serializers
from model import Generator, MSDiscriminator
from dataset import DatasetLoader
from utils import set_optimizer
from pathlib import Path

xp = cuda.cupy
cuda.get_device(0).use()


class CycleGANVC2LossFunction:
    def __init__(self):
        pass

    def adv_dis_loss(self, discriminator, y, t):
        fake_list = discriminator(y)
        real_list = discriminator(t)

        sum_loss = 0

        for fake, real in zip(fake_list, real_list):
            loss = F.mean(F.softplus(-real)) + F.mea'n(F.softplus(fake))
            sum_loss += loss

        return sum_loss

    def adv_gen_loss(self, discriminator, y):
        fake_list = discriminator(y)

        sum_loss = 0

        for fake in fake_list:
            loss = F.mean(F.softplus(-fake))
            sum_loss += loss

        return sum_loss

    def recon_loss(self, y, t):
        return F.mean_absolute_error(y, t)


def train(epochs, iterations, batchsize, src_path, tgt_path, modeldir):
    # Dataset definition
    dataset = DatasetLoader(src_path, tgt_path)
    print(dataset)

    # Model & Optimizer Definition
    generator_xy = Generator()
    generator_xy.to_gpu()
    gen_xy_opt = set_optimizer(generator_xy)

    generator_yx = Generator()
    generator_yx.to_gpu()
    gen_yx_opt = set_optimizer(generator_yx)

    discriminator_y = MSDiscriminator()
    discriminator_y.to_gpu()
    dis_y_opt = set_optimizer(discriminator_y)

    discriminator_x = MSDiscriminator()
    discriminator_x.to_gpu()
    dis_x_opt = set_optimizer(discriminator_x)

    # Loss Function Definition
    lossfunc = CycleGANVC2LossFunction()

    for epoch in range(epochs):
        sum_gen_loss = 0
        sum_dis_loss = 0
        for batch in range(0, iterations, batchsize):
            x, y = dataset.train(batchsize)

            xy = generator_xy(x)
            yx = generator_yx(y)

            xy.unchain_backward()
            yx.unchain_backward()

            loss = lossfunc.adv_dis_loss(discriminator_y, xy, y)
            loss += lossfunc.adv_dis_loss(discriminator_x, yx, x)

            sum_dis_loss += loss.data

            discriminator_x.cleargrads()
            discriminator_y.cleargrads()
            loss.backward()
            dis_x_opt.update()
            dis_y_opt.update()
            loss.unchain_backward()

            xy = generator_xy(x)
            xyx = generator_yx(xy)
            id_y = generator_xy(y)

            yx = generator_yx(y)
            yxy = generator_xy(yx)
            id_x = generator_yx(x)

            loss = lossfunc.adv_gen_loss(discriminator_y, xy)
            loss += lossfunc.adv_gen_loss(discriminator_x, yx)

            cycle_loss_x = lossfunc.recon_loss(xyx, x)
            cycle_loss_y = lossfunc.recon_loss(yxy, y)
            cycle_loss = cycle_loss_x + cycle_loss_y

            identity_loss_x = lossfunc.recon_loss(id_y, y)
            identity_loss_y = lossfunc.recon_loss(id_x, x)
            identity_loss = identity_loss_x + identity_loss_y

            if epoch > 20:
                identity_weight = 0.0
            else:
                identity_weight = 5.0

            loss += 10 * cycle_loss + identity_weight * identity_loss

            generator_xy.cleargrads()
            generator_yx.cleargrads()
            loss.backward()
            gen_xy_opt.update()
            gen_yx_opt.update()
            loss.unchain_backward()

            sum_gen_loss += loss.data.get()

            if batch == 0:
                serializers.save_npz(f"{modeldir}/generator_xy.model", generator_xy)
                serializers.save_npz(f"{modeldir}/generator_yx.model", generator_yx)

        print('epoch : {}'.format(epoch))
        print('Generator loss : {}'.format(sum_gen_loss / iterations))
        print('Discriminator loss : {}'.format(sum_dis_loss / iterations))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CycleGANVC2")
    parser.add_argument("--e", type=int, default=1000, help="the number of epochs")
    parser.add_argument("--i", type=int, default=2000, help="the number of iterations")
    parser.add_argument("--b", type=int, default=16, help="batch size")

    args = parser.parse_args()

    src_path = Path("./jsut_ver1.1/basic5000/wav/")
    tgt_path = Path("./ayanami/")

    modeldir = Path("./modeldir")
    modeldir.mkdir(exist_ok=True)

    train(args.e, args.i, args.b, src_path, tgt_path, modeldir)

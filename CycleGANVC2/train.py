import chainer
import chainer.functions as F
import numpy as np
import argparse

from model import Generator, Discriminator
from chainer import cuda, serializers
from pathlib import Path
from utils import set_optimizer
from dataset import DatasetLoader

xp = cuda.cupy
cuda.get_device(0).use()


class CycleGANVC2LossCalculator:
    def __init__(self):
        pass

    @staticmethod
    def dis_loss(discriminator, y, t):
        y_dis = discriminator(y)
        t_dis = discriminator(t)

        return F.mean(F.softplus(-t_dis)) + F.mean(F.softplus(y_dis))

    @staticmethod
    def gen_loss(discriminator, y):
        y_dis = discriminator(y)

        return F.mean(F.softplus(-y_dis))

    @staticmethod
    def cycle_loss(y, t):
        return 10.0 * F.mean_absolute_error(y, t)

    @staticmethod
    def identity_loss(y, t):
        return 5.0 * F.mean_absolute_error(y, t)


def train(epochs,
          iterations,
          batchsize,
          modeldir,
          extension,
          time_width,
          mel_bins,
          sampling_rate,
          g_learning_rate,
          d_learning_rate,
          beta1,
          beta2,
          identity_epoch,
          second_step,
          src_path,
          tgt_path):

    # Dataset definiton
    dataset = DatasetLoader(src_path,
                            tgt_path,
                            extension,
                            time_width,
                            mel_bins,
                            sampling_rate)
    print(dataset)

    # Model & Optimizer definition
    generator_xy = Generator()
    generator_xy.to_gpu()
    gen_xy_opt = set_optimizer(generator_xy, g_learning_rate, beta1, beta2)

    generator_yx = Generator()
    generator_yx.to_gpu()
    gen_yx_opt = set_optimizer(generator_yx, g_learning_rate, beta1, beta2)

    discriminator_y = Discriminator()
    discriminator_y.to_gpu()
    dis_y_opt = set_optimizer(discriminator_y, d_learning_rate, beta1, beta2)

    discriminator_x = Discriminator()
    discriminator_x.to_gpu()
    dis_x_opt = set_optimizer(discriminator_x, d_learning_rate, beta1, beta2)

    discriminator_xyx = Discriminator()
    discriminator_xyx.to_gpu()
    dis_xyx_opt = set_optimizer(discriminator_xyx, d_learning_rate, beta1, beta2)

    discriminator_yxy = Discriminator()
    discriminator_yxy.to_gpu()
    dis_yxy_opt = set_optimizer(discriminator_yxy, d_learning_rate, beta1, beta2)

    # Loss function definition
    lossfunc = CycleGANVC2LossCalculator()

    for epoch in range(epochs):
        sum_dis_loss = 0
        sum_gen_loss = 0

        for batch in range(0, iterations, batchsize):
            x, y = dataset.train(batchsize)

            xy = generator_xy(x)
            xyx = generator_yx(xy)

            yx = generator_yx(y)
            yxy = generator_xy(yx)

            xy.unchain_backward()
            xyx.unchain_backward()
            yx.unchain_backward()
            yxy.unchain_backward()

            dis_loss = lossfunc.dis_loss(discriminator_y, xy, y)
            dis_loss += lossfunc.dis_loss(discriminator_x, yx, x)

            if second_step:
                dis_loss += lossfunc.dis_loss(discriminator_xyx, xyx, x)
                dis_loss += lossfunc.dis_loss(discriminator_yxy, yxy, y)

                discriminator_xyx.cleargrads()
                discriminator_yxy.cleargrads()

            discriminator_x.cleargrads()
            discriminator_y.cleargrads()

            dis_loss.backward()
            dis_x_opt.update()
            dis_y_opt.update()

            if second_step:
                dis_xyx_opt.update()
                dis_yxy_opt.update()

            dis_loss.unchain_backward()

            xy = generator_xy(x)
            xyx = generator_yx(xy)
            id_y = generator_xy(y)

            yx = generator_yx(y)
            yxy = generator_xy(yx)
            id_x = generator_yx(x)

            gen_loss = lossfunc.gen_loss(discriminator_y, xy)
            gen_loss += lossfunc.gen_loss(discriminator_x, yx)

            if second_step:
                gen_loss += lossfunc.gen_loss(discriminator_yxy, yxy)
                gen_loss += lossfunc.gen_loss(discriminator_xyx, xyx)

            gen_loss += lossfunc.cycle_loss(x, xyx)
            gen_loss += lossfunc.cycle_loss(y, xyx)

            if epoch < identity_epoch:
                gen_loss += lossfunc.identity_loss(id_y, y)
                gen_loss += lossfunc.identity_loss(id_x, x)

            generator_xy.cleargrads()
            generator_yx.cleargrads()
            gen_loss.backward()
            gen_xy_opt.update()
            gen_yx_opt.update()
            gen_loss.unchain_backward()

            sum_dis_loss += dis_loss.data
            sum_gen_loss += gen_loss.data

            if batch == 0:
                serializers.save_npz(f"{modeldir}/generator_xy_{epoch}.model", generator_xy)
                serializers.save_npz(f"{modeldir}/generator_yx_{epoch}.model", generator_yx)

        print('epoch : {}'.format(epoch))
        print('Generator loss : {}'.format(sum_gen_loss / iterations))
        print('Discriminator loss : {}'.format(sum_dis_loss / iterations))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="StarGANVC2")
    parser.add_argument('--e', type=int, default=50, help="the number of epochs")
    parser.add_argument('--i', type=int, default=1000, help="the number of iterations")
    parser.add_argument('--b', type=int, default=16, help="batch size")
    parser.add_argument('--modeldir', type=Path, default="modeldir", help="model output directory")
    parser.add_argument('--ext', type=str, default=".npy", help="extension of training data")
    parser.add_argument('--tw', type=int, default=128, help="time width of spectral envelope")
    parser.add_argument('--mb', type=int, default=36, help="mel bins of spectral envelope")
    parser.add_argument('--sr', type=int, default=22050, help="sampling rate of audio data")
    parser.add_argument('--glr', type=float, default=0.0002, help="learning rate of Adam on generator")
    parser.add_argument('--dlr', type=float, default=0.0001, help="learning rate of Adam on discriminator")
    parser.add_argument('--b1', type=float, default=0.5, help="beta1 of Adam")
    parser.add_argument('--b2', type=float, default=0.999, help="beta2 of Adam")
    parser.add_argument('--ie', type=int, default=20, help="time spans enabling identity mapping loss")
    parser.add_argument('--second', action="store_true", help="enabling second step of adversaria loss")
    parser.add_argument('--src', type=Path, help="path which includes source data")
    parser.add_argument('--tgt', type=Path, help="path which includes target data")
    args = parser.parse_args()

    modeldir = args.modeldir
    modeldir.mkdir(exist_ok=True)

    train(args.e, args.i, args.b, modeldir, args.ext, args.tw, args.mb, args.sr,
          args.glr, args.dlr, args.b1, args.b2, args.ie, args.second,
          args.src, args.tgt)

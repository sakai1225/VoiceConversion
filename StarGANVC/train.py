import chainer.functions as F
import argparse
import chainer

from model import Generator, Discriminator, GeneratorWithCIN
from dataset import DatasetLoader
from utils import set_optimizer, call_zeros
from pathlib import Path
from chainer import cuda, serializers

xp = cuda.cupy
cuda.get_device(0).use()


class StarGANVC2LossFunction:
    def __init__(self):
        pass

    def dis_loss(self, discriminator, y_fake, x, y_label, x_label, residual):
        x = chainer.Variable(x.data)

        if residual:
            fake = discriminator(y_fake + x, y_label)
        else:
            fake = discriminator(y_fake, y_label)

        real = discriminator(x, x_label)

        return F.mean(F.softplus(-real)), F.mean(F.softplus(fake))

    @staticmethod
    def gen_loss(discriminator, y_fake, x_fake, x, y_label, residual):
        if residual:
            fake = discriminator(y_fake + x, y_label)
        else:
            fake = discriminator(y_fake, y_label)

        loss = F.mean(F.softplus(-fake))

        return loss, 10 * F.mean_absolute_error(x_fake, x)

    @staticmethod
    def identity_loss(x_identity, x):
        return 5.0 * F.mean_absolute_error(x_identity, x)


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
          adv_type,
          residual_flag,
          data_path):

    # Dataset Definition
    dataloader = DatasetLoader(data_path)

    # Model & Optimizer Definition
    generator = GeneratorWithCIN(adv_type=adv_type)
    generator.to_gpu()
    gen_opt = set_optimizer(generator, g_learning_rate, beta1, beta2)

    discriminator = Discriminator()
    discriminator.to_gpu()
    dis_opt = set_optimizer(discriminator, d_learning_rate, beta1, beta2)

    # Loss Function Definition
    lossfunc = StarGANVC2LossFunction()

    for epoch in range(epochs):
        sum_dis_loss = 0
        sum_gen_loss = 0
        for batch in range(0, iterations, batchsize):
            x_sp, x_label, y_sp, y_label = dataloader.train(batchsize)

            if adv_type == 'sat':
                y_fake = generator(x_sp, F.concat([y_label, x_label]))
            elif adv_type == 'orig':
                y_fake = generator(x_sp, y_label)
            else:
                raise AttributeError

            y_fake.unchain_backward()

            if adv_type == 'sat':
                advloss_dis_real, advloss_dis_fake = lossfunc.dis_loss(
                    discriminator,
                    y_fake,
                    x_sp,
                    F.concat([y_label, x_label]),
                    F.concat([x_label, y_label]),
                    residual_flag
                )
            elif adv_type == 'orig':
                advloss_dis_real, advloss_dis_fake = lossfunc.dis_loss(
                    discriminator,
                    y_fake,
                    x_sp,
                    y_label,
                    x_label,
                    residual_flag
                )
            else:
                raise AttributeError

            dis_loss = advloss_dis_real + advloss_dis_fake
            discriminator.cleargrads()
            dis_loss.backward()
            dis_opt.update()
            dis_loss.unchain_backward()

            if adv_type == 'sat':
                y_fake = generator(x_sp, F.concat([y_label, x_label]))
                x_fake = generator(y_fake, F.concat([x_label, y_label]))
                x_identity = generator(x_sp, F.concat([x_label, x_label]))
                advloss_gen_fake, cycle_loss = lossfunc.gen_loss(
                    discriminator,
                    y_fake,
                    x_fake,
                    x_sp,
                    F.concat([y_label, x_label]),
                    residual_flag
                )
            elif adv_type == 'orig':
                y_fake = generator(x_sp, y_label)
                x_fake = generator(y_fake, x_label)
                x_identity = generator(x_sp, x_label)
                advloss_gen_fake, cycle_loss = lossfunc.gen_loss(
                    discriminator,
                    y_fake,
                    x_fake,
                    x_sp,
                    y_label,
                    residual_flag
                )
            else:
                raise AttributeError

            if epoch < identity_epoch:
                identity_loss = lossfunc.identity_loss(x_identity, x_sp)
            else:
                identity_loss = call_zeros(advloss_dis_fake)

            gen_loss = advloss_gen_fake + cycle_loss + identity_loss
            generator.cleargrads()
            gen_loss.backward()
            gen_opt.update()
            gen_loss.unchain_backward()

            sum_dis_loss += dis_loss.data
            sum_gen_loss += gen_loss.data

            if batch == 0:
                serializers.save_npz(f"{modeldir}/generator_{epoch}.model", generator)

        print(f"epoch: {epoch}")
        print(f"dis loss: {sum_dis_loss / iterations} gen loss: {sum_gen_loss / iterations}")


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
    parser.add_argument('--res', action="store_true", help="enable residual connenction")
    parser.add_argument('--adv_type', type=str, default='sat', help="the type of adversarial loss")
    parser.add_argument('--path', type=Path, help="path which includes training data")
    args = parser.parse_args()

    modeldir = args.modeldir
    modeldir.mkdir(exist_ok=True)

    train(args.e, args.i, args.b, modeldir, args.ext, args.tw, args.mb, args.sr,
          args.glr, args.dlr, args.b1, args.b2, args.ie, args.adv_type, args.res, args.path)

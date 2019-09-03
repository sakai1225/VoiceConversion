import chainer
import chainer.functions as F
import argparse

from model import Generator, Discriminator, GeneratorWithCIN
from dataset import DatasetLoader
from utils import set_optimizer
from pathlib import Path
from chainer import cuda, serializers

xp = cuda.cupy
cuda.get_device(0).use()


class StarGANVC2LossFunction:
    def __init__(self):
        pass

    @staticmethod
    def dis_loss(discriminator, y_fake, x, y_label, x_label):
        fake = discriminator(y_fake, F.concat([y_label, x_label]))
        real = discriminator(x, F.concat([x_label, y_label]))
        #fake = discriminator(y_fake, y_label)
        #"real = discriminator(x, x_label)

        return F.mean(F.softplus(-real)) + F.mean(F.softplus(fake))

    @staticmethod
    def gen_loss(discriminator, y_fake, x_fake, x, y_label):
        fake = discriminator(y_fake, y_label)

        loss = F.mean(F.softplus(-fake))
        loss += 10 * F.mean_absolute_error(x_fake, x)

        return loss

    @staticmethod
    def identity_loss(x_identity, x):
        return 5.0 * F.mean_absolute_error(x_identity, x)


def train(epochs, iterations, batchsize, outdir, data_path):
    # Dataset Definition
    dataloader = DatasetLoader(data_path)

    # Model & Optimizer Definition
    #generator = Generator()
    generator = GeneratorWithCIN()
    generator.to_gpu()
    gen_opt = set_optimizer(generator, alpha=0.0002)

    discriminator = Discriminator()
    discriminator.to_gpu()
    dis_opt = set_optimizer(discriminator, alpha=0.0001)

    # Loss Function Definition
    lossfunc = StarGANVC2LossFunction()

    for epoch in range(epochs):
        sum_loss = 0
        for batch in range(0, iterations, batchsize):
            x_sp, x_label, y_sp, y_label = dataloader.train(batchsize)
            y_fake = generator(x_sp, F.concat([y_label, x_label]))
            y_fake.unchain_backward()

            loss = lossfunc.dis_loss(discriminator, y_fake, x_sp, y_label, x_label)

            discriminator.cleargrads()
            loss.backward()
            dis_opt.update()
            loss.unchain_backward()

            y_fake = generator(x_sp, F.concat([y_label, x_label]))
            x_fake = generator(y_fake, F.concat([x_label, y_label]))
            x_identity = generator(x_sp, F.concat([x_label, x_label]))
            loss = lossfunc.gen_loss(discriminator, y_fake, x_fake, x_sp, F.concat([y_label, x_label]))
            if epoch < 50:
                loss += lossfunc.identity_loss(x_identity, x_sp)

            generator.cleargrads()
            loss.backward()
            gen_opt.update()
            loss.unchain_backward()

            sum_loss += loss.data

            if batch == 0:
                serializers.save_npz(f"modeldirCIN/generator_{epoch}.model", generator)
                serializers.save_npz('discriminator.model', discriminator)

        print(f"epoch: {epoch}")
        print(f"loss: {sum_loss / iterations}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="StarGANVC2")
    parser.add_argument('--e', type=int, default=1000, help="the number of epochs")
    parser.add_argument('--i', type=int, default=10000, help="the number of iterations")
    parser.add_argument('--b', type=int, default=16, help="batch size")
    args = parser.parse_args()

    data_path = Path('./starganvc/')
    outdir = Path('outdir')
    outdir.mkdir(exist_ok=True)

    train(args.e, args.i, args.b, outdir, data_path)
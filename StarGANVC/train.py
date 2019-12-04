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

    @staticmethod
    def _zero_centered_gradient_penalty(y_dis, y):
        grad, = chainer.grad([y_dis], [y], enable_double_backprop=True)
        grad = F.sqrt(F.batch_l2_norm_squared(grad))
        zeros = call_zeros(grad)

        loss = 10 * F.mean_squared_error(grad, zeros)

        return loss

    def dis_loss(self, discriminator, y_fake, x, y_label, x_label, residual):
        x = chainer.Variable(x.data)

        if residual:
            fake = discriminator(y_fake + x, y_label)
        else:
            fake = discriminator(y_fake, y_label)

        real = discriminator(x, x_label)

        #loss = self._zero_centered_gradient_penalty(fake, y_fake)
        #loss += self._zero_centered_gradient_penalty(real, x)

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


def train(epochs, iterations, batchsize, adv_type, modeldir, data_path, residual):
    # Dataset Definition
    dataloader = DatasetLoader(data_path)

    # Model & Optimizer Definition
    #generator = Generator()
    generator = GeneratorWithCIN(adv_type=adv_type)
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
                    residual
                )
            elif adv_type == 'orig':
                advloss_dis_real, advloss_dis_fake = lossfunc.dis_loss(
                    discriminator,
                    y_fake,
                    x_sp,
                    y_label,
                    x_label,
                    residual
                )
            else:
                raise AttributeError

            loss = advloss_dis_real + advloss_dis_fake
            discriminator.cleargrads()
            loss.backward()
            dis_opt.update()
            loss.unchain_backward()

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
                    residual
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
                    residual
                )
            else:
                raise AttributeError

            if epoch < 20:
                identity_loss = lossfunc.identity_loss(x_identity, x_sp)
            else:
                identity_loss = call_zeros(advloss_dis_fake)

            loss = advloss_gen_fake + cycle_loss + identity_loss
            generator.cleargrads()
            loss.backward()
            gen_opt.update()
            loss.unchain_backward()

            sum_loss += loss.data

            if batch == 0:
                serializers.save_npz(f"{modeldir}/generator_{epoch}.model", generator)
                serializers.save_npz('discriminator.model', discriminator)

        print(f"epoch: {epoch}")
        print(f"loss: {sum_loss / iterations}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="StarGANVC2")
    parser.add_argument('--e', type=int, default=50, help="the number of epochs")
    parser.add_argument('--i', type=int, default=10000, help="the number of iterations")
    parser.add_argument('--b', type=int, default=16, help="batch size")
    parser.add_argument('--res', action="stroe_true", help="enable residual connenction")
    parser.add_argument('--adv_type', type=str, default='sat', help="the type of adversarial loss")
    args = parser.parse_args()

    data_path = Path('./starganvc_sp/')
    modeldir = Path('modeldirCIN_res')
    modeldir.mkdir(exist_ok=True)

    train(args.e, args.i, args.b, args.adv_type, modeldir, data_path, args.res)

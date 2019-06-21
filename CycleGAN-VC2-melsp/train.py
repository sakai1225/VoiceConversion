import chainer
import chainer.links as L
import chainer.functions as F
from chainer import cuda, Chain, optimizers, serializers
import numpy as np
import os, glob
import argparse
import matplotlib
matplotlib.use('Agg')
import pylab
import librosa
from model import Generator, Discriminator, x4_Discriminator
from lain.audio.navi import load, save
from lain.audio.layer.converter import audio2af, encode_sp, audio2melsp
from lain.audio.layer.augmentation import random_crop
from librosa.display import specshow

xp = cuda.cupy
cuda.get_device(0).use()

def set_optimizer(model, alpha=0.0002, beta1=0.5):
    optimizer = optimizers.Adam(alpha=alpha, beta1=beta1)
    optimizer.setup(model)

    return optimizer

def normalize(x, epsilon=1e-8):
    x = librosa.power_to_db(x, ref=np.max)
    x = (x + 80) / 80

    return x

def crop(sp, upper_bound=128):
    if sp.shape[0] < upper_bound + 1:
        sp = np.pad(sp, ((0, upper_bound-sp.shape[0] + 2), (0, 0)), 'constant', constant_values=0)

    start_point = np.random.randint(sp.shape[0] - upper_bound)
    cropped = sp[start_point : start_point + upper_bound, :]

    return cropped

def x4_downsampling(x):
    h = F.average_pooling_2d(x, 3, 2, 1)
    h = F.average_pooling_2d(x, 3, 2, 1)

    return h

parser = argparse.ArgumentParser(description='CycleGAN-VC2')
parser.add_argument('--epoch', default=1000, type=int, help="the number of epochs")
parser.add_argument('--batchsize', default=16, type=int, help="batch size")
parser.add_argument('--testsize', default=2, type=int, help="test size")
parser.add_argument('--Ntrain', default=5000, type=int, help="data size")
parser.add_argument('--cw', default=10.0, type=float, help="the weight of cycle loss")
parser.add_argument('--iw', default=5.0, type=float, help="the weight of identity loss")

args = parser.parse_args()
epochs = args.epoch
batchsize = args.batchsize
testsize = args.testsize
Ntrain = args.Ntrain
cycle_weight = args.cw
identity_weight = args.iw

x_path = './jsut_ver1.1/basic5000/wav/'
#x_f0_path = '/data/users/hasegawa/Dataset/jsut_ver1.1/basic5000/f0/'
x_list = os.listdir(x_path)
x_len = len(x_list)
y_path = './ayanami/'
#y_f0_path = '/data/users/hasegawa/Dataset/kaede_f0/'
y_list = glob.glob(y_path + '*/*/*.wav', recursive=True)
y_len = len(y_list)

generator_xy = Generator()
generator_xy.to_gpu()
gen_xy_opt = set_optimizer(generator_xy)

generator_yx = Generator()
generator_yx.to_gpu()
gen_yx_opt = set_optimizer(generator_yx)

discriminator_y = Discriminator()
discriminator_y.to_gpu()
dis_y_opt = set_optimizer(discriminator_y, alpha=0.0001)

discriminator_x = Discriminator()
discriminator_x.to_gpu()
dis_x_opt = set_optimizer(discriminator_x, alpha=0.0001)

x4_discriminator_y = x4_Discriminator()
x4_discriminator_y.to_gpu()
x4dis_y_opt = set_optimizer(x4_discriminator_y, alpha=0.0001)

x4_discriminator_x = x4_Discriminator()
x4_discriminator_x.to_gpu()
x4dis_x_opt = set_optimizer(x4_discriminator_x, alpha=0.0001)

for epoch in range(epochs):
    sum_gen_loss = 0
    sum_dis_loss = 0
    for batch in range(0, Ntrain, batchsize):
        x_melsp_box = []
        y_melsp_box = []
        for _ in range(batchsize):
            rnd_x = np.random.randint(x_len)
            ad_x = load(x_path + x_list[rnd_x], sampling_rate=22050)
            melsp_x = normalize(audio2melsp(ad_x).transpose(1, 0))
            rnd_y = np.random.randint(y_len)
            ad_y = load(y_list[rnd_y], sampling_rate=22050)
            melsp_y = normalize(audio2melsp(ad_y).transpose(1, 0))
            sp_x = crop(melsp_x, upper_bound=128)
            sp_y = crop(melsp_y, upper_bound=128)
            #_, sp_x, _ = audio2af(xx)
            #_, sp_y, _ = audio2af(yy)

            x_melsp_box.append(sp_x[np.newaxis, :])
            y_melsp_box.append(sp_y[np.newaxis, :])

        x = chainer.as_variable(xp.array(x_melsp_box).astype(xp.float32))
        y = chainer.as_variable(xp.array(y_melsp_box).astype(xp.float32))

        xy = generator_xy(x)
        xyx = generator_yx(xy)

        yx = generator_yx(y)
        yxy = generator_xy(yx)

        xy.unchain_backward()
        xyx.unchain_backward()
        yx.unchain_backward()
        yxy.unchain_backward()

        dis_y_fake = discriminator_y(xy)
        dis_y_real = discriminator_y(y)
        dis_x_fake = discriminator_x(yx)
        dis_x_real = discriminator_x(x)

        x4dis_y_fake = x4_discriminator_y(x4_downsampling(xy))
        x4dis_y_real = x4_discriminator_y(x4_downsampling(y))
        x4dis_x_fake = x4_discriminator_x(x4_downsampling(yx))
        x4dis_x_real = x4_discriminator_x(x4_downsampling(x))

        dis_loss = F.mean(F.softplus(dis_y_fake)) + F.mean(F.softplus(-dis_y_real))
        dis_loss += F.mean(F.softplus(dis_x_fake)) + F.mean(F.softplus(-dis_x_real))
        dis_loss += F.mean(F.softplus(x4dis_x_fake)) + F.mean(F.softplus(-x4dis_x_real))
        dis_loss += F.mean(F.softplus(x4dis_y_fake)) + F.mean(F.softplus(-x4dis_y_real))

        discriminator_x.cleargrads()
        discriminator_y.cleargrads()
        x4_discriminator_x.cleargrads()
        x4_discriminator_y.cleargrads()
        dis_loss.backward()
        dis_x_opt.update()
        dis_y_opt.update()
        x4dis_x_opt.update()
        x4dis_y_opt.update()
        dis_loss.unchain_backward()

        xy = generator_xy(x)
        xyx = generator_yx(xy)
        id_y = generator_xy(y)

        yx = generator_yx(y)
        yxy = generator_xy(yx)
        id_x = generator_yx(x)

        dis_y_fake = discriminator_y(xy)
        dis_x_fake = discriminator_x(yx)
        x4dis_y_fake = x4_discriminator_y(xy)
        x4dis_x_fake = x4_discriminator_x(yx)

        cycle_loss_x = F.mean_absolute_error(x, xyx)
        cycle_loss_y = F.mean_absolute_error(y, yxy)
        cycle_loss = cycle_loss_x + cycle_loss_y
        
        identity_loss_x = F.mean_absolute_error(id_y, y)
        identity_loss_y = F.mean_absolute_error(id_x, x)
        identity_loss = identity_loss_x + identity_loss_y

        if epoch > 20:
            identity_weight = 0.0
        
        gen_loss = F.mean(F.softplus(-dis_x_fake)) + F.mean(F.softplus(-dis_y_fake))
        gen_loss += F.mean(F.softplus(-x4dis_x_fake)) + F.mean(F.softplus(-x4dis_y_fake))
        gen_loss += cycle_weight * cycle_loss + identity_weight * identity_loss

        generator_xy.cleargrads()
        generator_yx.cleargrads()
        gen_loss.backward()
        gen_xy_opt.update()
        gen_yx_opt.update()
        gen_loss.unchain_backward()

        sum_dis_loss += dis_loss.data.get()
        sum_gen_loss += gen_loss.data.get()

        if batch == 0:
            serializers.save_npz('generator_xy.model', generator_xy)
            serializers.save_npz('generator_yx.model', generator_yx)
            #pylab.imshow(xy[0][0].data.transpose(1, 0), aspect='auto', origin='bottom', interpolation='none')
            #pylab.savefig('./melsp_train.png')

    print('epoch : {}'.format(epoch))
    print('Generator loss : {}'.format(sum_gen_loss / x_len))
    print('Discriminator loss : {}'.format(sum_dis_loss / x_len))
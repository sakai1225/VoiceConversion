import chainer
import chainer.links as L
import chainer.functions as F
from chainer import cuda, Chain, optimizers, serializers
import numpy as np
import os
import argparse
import pylab
from model import Generator, ContentDiscriminator, DomainDiscriminator
from lain.audio.navi import load, save
from lain.audio.layer.converter import audio2af, encode_sp
from lain.audio.layer.augmentation import random_crop
from librosa.display import specshow

xp = cuda.cupy
cuda.get_device(0).use()

def set_optimizer(model, alpha=0.0002, beta1=0.5):
    """Adam optimizer setup
    
    Args:
        model : model paramter
        alpha (float, optional): Defaults to 0.0002. the alpha parameter of Adam
        beta1 (float, optional): Defaults to 0.5. the beta1 parameter of Adam
    """
    optimizer = optimizers.Adam(alpha=alpha, beta1=beta1)
    optimizer.setup(model)

    return optimizer

def normalize(x):
    """normalization of spectral envelope. mean 0, variance 1
    
    Args:
        x (numpy.float): spectral envelope
        
    Returns:
        numpy.float: normalized spectral envelope
    """
    x_mean, x_std = np.mean(x, axis=1, keepdims=True), np.std(x, axis=1, keepdims=True)

    return (x - x_mean) / x_std

def crop(sp, upper_bound=128):
    """Cropping of spectral envelope
    
    Args:
        sp (numpy.float): spectral envelope
        upper_bound (int, optional): Defaults to 128. The size of cropping
    
    Returns:
        numpy.float: cropped spectral envelope
    """
    if sp.shape[0] < upper_bound + 1:
        sp = np.pad(sp, ((0, upper_bound-sp.shape[0] + 2), (0, 0)), 'constant', constant_values=0)

    start_point = np.random.randint(sp.shape[0] - upper_bound)
    cropped = sp[start_point : start_point + upper_bound, :]

    return cropped

def adversarial_content_D(discriminator, content_x, content_y):
    fake = discriminator(content_x)
    real = discriminator(content_y)

    ones = chainer.as_variable(xp.ones((real.shape[0], 1, real.shape[2], real.shape[3])).astype(xp.int32))
    zeros = chainer.as_variable(xp.zeros((fake.shape[0], 1, fake.shape[2],  fake.shape[3])).astype(xp.int32))

    loss = F.sigmoid_cross_entropy(real ,ones) + F.sigmoid_cross_entropy(fake, zeros)

    return loss

def adversarial_content_G(discriminator, content_x, content_y):
    fake = discriminator(content_x)
    real = discriminator(content_y)

    ones = chainer.as_variable(xp.ones((fake.shape[0], 1, fake.shape[2], fake.shape[3])).astype(xp.int32))

    loss = (F.sigmoid_cross_entropy(real, ones) + F.sigmoid_cross_entropy(fake, ones)) / 2

    return loss

def adversarial_domain_D(discriminator, pred_fake, pred_real):
    fake = discriminator(pred_fake)
    real = discriminator(pred_real)

    return F.mean(F.softplus(fake)) + F.mean(F.softplus(-real))

def adversarial_domain_G(discriminator, pred_fake):
    fake = discriminator(pred_fake)

    return F.mean(F.softplus(-fake))

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

x_path = './Dataset/Speech/jsut_sp/'
x_list = os.listdir(x_path)
x_len = len(x_list)
y_path = './Dataset/Speech/ayanami_sp/'
y_list = os.listdir(y_path)
y_len = len(y_list)

generator = Generator()
generator.to_gpu()
gen_opt = set_optimizer(generator)

discriminator_y = DomainDiscriminator()
discriminator_y.to_gpu()
dis_y_opt = set_optimizer(discriminator_y, alpha=0.0001)

discriminator_x = DomainDiscriminator()
discriminator_x.to_gpu()
dis_x_opt = set_optimizer(discriminator_x, alpha=0.0001)

content_discriminator = ContentDiscriminator()
content_discriminator.to_gpu()
dis_con_opt = set_optimizer(content_discriminator, alpha=0.0001)

for epoch in range(epochs):
    sum_gen_loss = 0
    sum_dis_loss = 0
    for batch in range(0, Ntrain, batchsize):
        x_sp_box = []
        y_sp_box = []
        for _ in range(batchsize):
            # sp loading -> mel conversion -> normalization -> crop
            rnd_x = np.random.randint(x_len)
            sp_x = np.load(x_path + x_list[rnd_x])
            sp_x = normalize(encode_sp(sp_x, mel_bins=36))
            rnd_y = np.random.randint(y_len)
            sp_y = np.load(y_path + y_list[rnd_y])
            sp_y = normalize(encode_sp(sp_y, mel_bins=36))
            sp_x = crop(sp_x, upper_bound=128)
            sp_y = crop(sp_y, upper_bound=128)

            x_sp_box.append(sp_x[np.newaxis,:])
            y_sp_box.append(sp_y[np.newaxis,:])

        x = chainer.as_variable(xp.array(x_sp_box).astype(xp.float32))
        y = chainer.as_variable(xp.array(y_sp_box).astype(xp.float32))

        # Discriminator update
        a_out, b_out = generator(x, y)
        a_enc, _, a_fake, _, a_infer = a_out
        b_enc, _, b_fake, _,  b_infer = b_out
        _, a_infer, _ = a_infer
        _, b_infer, _ = b_infer

        a_enc.unchain_backward()
        a_fake.unchain_backward()
        a_infer.unchain_backward()
        b_enc.unchain_backward()
        b_fake.unchain_backward()
        b_infer.unchain_backward()

        dis_loss = adversarial_content_D(content_discriminator, a_enc, b_enc)
        dis_loss += adversarial_domain_D(discriminator_x, a_fake, x)
        dis_loss += adversarial_domain_D(discriminator_y, b_fake, y)
        dis_loss += adversarial_domain_D(discriminator_x, a_infer, x)
        dis_loss += adversarial_domain_D(discriminator_y, a_infer, y)

        content_discriminator.cleargrads()
        discriminator_x.cleargrads()
        discriminator_y.cleargrads()
        dis_loss.backward()
        dis_con_opt.update()
        dis_x_opt.update()
        dis_y_opt.update()
        dis_loss.unchain_backward()

        # Generator update
        a_out, b_out = generator(x, y)
        a_enc, a_attr, a_fake, a_recon, a_infer = a_out
        b_enc, b_attr, b_fake, b_recon, b_infer = b_out
        a_latent, a_infer, a_infer_attr = a_infer
        b_latent, b_infer, b_infer_attr = b_infer
        a_out, b_out = generator(a_fake, b_fake)
        _, _, aba_fake, _, _ = a_out
        _, _, bab_fake, _, _ = b_out

        gen_loss = adversarial_content_G(content_discriminator, a_enc, b_enc)
        gen_loss += adversarial_domain_G(discriminator_x, a_fake)
        gen_loss += adversarial_domain_G(discriminator_y, b_fake)
        gen_loss += adversarial_domain_G(discriminator_x, a_infer)
        gen_loss += adversarial_domain_G(discriminator_y, b_infer)
        gen_loss += cycle_weight * F.mean_absolute_error(aba_fake, x)
        gen_loss += cycle_weight * F.mean_absolute_error(bab_fake, y)
        gen_loss += cycle_weight * F.mean_absolute_error(a_recon, x)
        gen_loss += cycle_weight * F.mean_absolute_error(b_recon, y)
        gen_loss += cycle_weight * F.mean_absolute_error(a_latent, a_infer_attr)
        gen_loss += cycle_weight * F.mean_absolute_error(b_latent, b_infer_attr)

        generator.cleargrads()
        gen_loss.backward()
        gen_opt.update()
        gen_loss.unchain_backward()

        sum_dis_loss += dis_loss.data.get()
        sum_gen_loss += gen_loss.data.get()

        if batch == 0:
            serializers.save_npz('./generator.model', generator)

    print('epoch : {}'.format(epoch))
    print('Generator loss : {}'.format(sum_gen_loss / x_len))
    print('Discriminator loss : {}'.format(sum_dis_loss / x_len))
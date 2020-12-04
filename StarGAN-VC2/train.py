import torch as pt
import torch.optim as opt
from torch.utils.tensorboard import SummaryWriter
import torch.nn as nn
import argparse

import numpy as  np

from model import Generator, Discriminator
from dataset import AudioCollator, AudioDataset
from pathlib import Path
from tqdm import tqdm
from torch.utils.data import DataLoader

softplus = nn.Softplus()
l1loss = nn.L1Loss()


def adversarial_loss_dis(discriminator, y, x, x_to_y, y_to_x):
    fake = discriminator(y.detach(), x_to_y)
    real = discriminator(x, y_to_x)

    return pt.mean(softplus(-real)), pt.mean(softplus(fake))


def adversarial_loss_gen(discriminator, y, x_to_y):
    fake = discriminator(y, x_to_y)

    return pt.mean(softplus(-fake))


def cycle_consistency_loss(y, t):
    return 5.0 * l1loss(y, t)


def identity_mapping_loss(y, t):
    return 10.0 * l1loss(y, t)


def write(writer, loss_kind, loss, iteration):
    writer.add_scalar(loss_kind, loss.item(), iteration)


def train(iteration, batchsize, data_path, data_dir, modeldir, cls_num, duration):
    # Dataset definition
    dataset = AudioDataset(data_path, data_dir)

    collator = AudioCollator(cls_num)

    # Model & Optimizer definition
    generator = Generator(cls_num=cls_num)
    generator.cuda()
    generator.train()
    gen_opt = opt.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))

    discriminator = Discriminator(cls_num)
    discriminator.cuda()
    discriminator.train()
    dis_opt = opt.Adam(discriminator.parameters(), lr=0.0001, betas=(0.5, 0.999))

    # Writer definition
    writer = SummaryWriter()

    iterations = 0

    if iterations != 0:
        pt.load(f"{modeldir}/generator_{iterations}.model", map_location=lambda storage, loc: storage.cuda(0))
        pt.load(f"{modeldir}/discriminator_{iterations}.model", map_location=lambda storage, loc: storage.cuda(0))

    while True:

        dataloader = DataLoader(dataset,
                                                           batch_size=batchsize,
                                                           shuffle=True,
                                                           collate_fn=collator,
                                                           drop_last=True)
        dataloader = tqdm(dataloader)

        for i, data in enumerate(dataloader):

            x_sp, x_label, y_label = data
            x_to_y = pt.cat([y_label, x_label], dim=1)
            y_to_x = pt.cat([x_label, y_label], dim=1)
            x_to_x = pt.cat([x_label, x_label], dim=1)

            # Discriminator update
            y_fake = generator(x_sp, x_to_y)

            # Adversarial loss
            dis_loss_real, dis_loss_fake = adversarial_loss_dis(discriminator, y_fake, x_sp, x_to_y, y_to_x)
            dis_loss = dis_loss_real + dis_loss_fake

            dis_opt.zero_grad()
            dis_loss.backward()
            dis_opt.step()

            write(writer, "dis_loss_real", dis_loss_real, iterations)
            write(writer, "dis_loss_fake", dis_loss_fake, iterations)

            # Generator update
            y_fake = generator(x_sp, x_to_y)
            x_fake = generator(y_fake, y_to_x)
            x_identity = generator(x_sp, x_to_x)

            # Adversarial loss
            gen_loss_fake = adversarial_loss_gen(discriminator, y_fake, x_to_y)

            # Cycle-consistency loss
            cycle_loss = cycle_consistency_loss(x_fake, x_sp)

            # # Identity-mapping loss
            # if epoch < duration:
            #     identity_loss = identity_mapping_loss(x_identity, x_sp)
            # else:
            #     identity_loss = pt.as_tensor(np.array(0))
            #     #identity_loss = pt.from_numpy(0)

            identity_loss = identity_mapping_loss(x_identity, x_sp)

            if iterations <= 10000:
                gen_loss = gen_loss_fake + cycle_loss + identity_loss
            else:
                gen_loss = gen_loss_fake + cycle_loss

            gen_opt.zero_grad()
            gen_loss.backward()
            gen_opt.step()

            write(writer, "gen_loss_fake", gen_loss_fake, iterations)
            write(writer, "cycle_loss", cycle_loss, iterations)
            write(writer, "identity_loss", identity_loss, iterations)

            print(f"iteration: {iterations}")
            print(f"dis loss real: {dis_loss_real} dis loss fake: {dis_loss_fake}")
            print(f"gen loss fake: {gen_loss_fake} cycle loss: {cycle_loss} identity loss: {identity_loss}")

            iterations += 1

            #save model after one epoch
            if iterations % 1000 == 0:
                pt.save(generator.state_dict(), f"{modeldir}/generator_{iterations}.model")
                pt.save(discriminator.state_dict(), f"{modeldir}/discriminator_{iterations}.model")

            # for loop break
            if iterations > iteration:
                break

        #while loop break
        if iterations > iteration:
            break




if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="StarGANVC2-pytorch")
    parser.add_argument("--e", type=int, default=30000, help="the number of epochs")
    parser.add_argument("--b", type=int, default=8, help="batch size")
    parser.add_argument("--n", type=int, default=4, help="the number of classes")
    parser.add_argument('--i', type=int, default=50, help="the duration of identity mapping loss")

    args = parser.parse_args()

    data_path = Path("./dataset")
    data_dir = ["SF1", "SF2", "TM1", "TM2"]
    modeldir = Path("E:/modeldir")
    modeldir.mkdir(exist_ok=True)

    train(args.e, args.b, data_path, data_dir, modeldir, args.n, args.i)
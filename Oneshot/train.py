import torch
import torch.nn as nn
import numpy as np
import argparse

from torch.utils.data import DataLoader
from pathlib import Path
from tqdm import tqdm
from dataset import JVSDataset, AudioCollate
from model import Model

l1loss = nn.L1Loss()


class OneshotLossCalculator:
    def __init__(self):
        pass

    @staticmethod
    def reconstruction_loss(y, t):
        return 10 * l1loss(y, t)

    @staticmethod
    def kl_loss(sigma, mu):
        return 0.5 * torch.mean(torch.exp(sigma) + mu ** 2 - 1 - sigma)


def train(epochs,
          interval,
          batchsize,
          modeldir,
          extension,
          time_width,
          learning_rate,
          beta1,
          beta2,
          weight_decay,
          annealing,
          kl_interval,
          data_path):

    # Dataset definition
    dataset = JVSDataset(data_path, extension)
    collator = AudioCollate(time_width)

    # Model definition
    model = Model(dim=512)
    model.cuda()
    model.train()
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=learning_rate,
                                 betas=(beta1, beta2),
                                 amsgrad=True,
                                 weight_decay=weight_decay)

    # Loss function definition
    lossfunc = OneshotLossCalculator()

    iteration = 0

    for epoch in range(epochs):
        dataloader = DataLoader(dataset,
                                batch_size=batchsize,
                                collate_fn=collator,
                                shuffle=True,
                                drop_last=True)
        dataloader = tqdm(dataloader)

        for i, data in enumerate(dataloader):
            iteration += 1
            x = data
            y, mu, sigma = model(x, x)
            loss = lossfunc.reconstruction_loss(y, x)

            if iteration % kl_interval == 0:
                if iteration < annealing:
                    kl_weight = 1.0 * iteration / annealing
                else:
                    kl_weight = 1.0

                loss += kl_weight * lossfunc.kl_loss(sigma, mu)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if iteration % interval == 1:
                torch.save(model.state_dict(), f"{modeldir}/model_{iteration}.pt")

            print(f"iteration: {iteration} loss: {loss.data}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Oneshot")
    parser.add_argument("--e", type=int, default=2000, help="the number of epochs")
    parser.add_argument("--i", type=int, default=2000, help="interval of snapshot")
    parser.add_argument("--b", type=int, default=192, help="batch size")
    parser.add_argument("--modeldir", type=Path, default="modeldir", help="model output directory")
    parser.add_argument("--ext", type=str, default=".npy", help="extension of training data")
    parser.add_argument("--tw", type=int, default=128, help="time width of spectral envelope")
    parser.add_argument("--lr", type=float, default=0.0005, help="learning rate of Adam")
    parser.add_argument("--b1", type=float, default=0.9, help="beta1 of Adam")
    parser.add_argument("--b2", type=float, default=0.999, help="beta2 of Adam")
    parser.add_argument("--wd", type=float, default=0.0001, help="weight decay")
    parser.add_argument("--a", type=int, default=50000, help="annealing iteration")
    parser.add_argument("--kli", type=int, default=1000, help="interval of kl loss")
    parser.add_argument("--path", type=Path, help="path which includes training data")

    args = parser.parse_args()

    modeldir = args.modeldir
    modeldir.mkdir(exist_ok=True)

    train(args.e, args.i, args.b, modeldir, args.ext, args.tw, args.lr,
          args.b1, args.b2, args.wd, args.a, args.kli, args.path)

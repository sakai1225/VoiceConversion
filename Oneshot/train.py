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


def reconstruction_loss(y, t):
    return 10 * l1loss(y, t)


def kl_loss(sigma, mu):
    return 0.01 * 0.5 * torch.mean(torch.exp(sigma) + mu ** 2 - 1 - sigma)


def train(epochs, batchsize, iterations, melpath, modeldir):
    # Dataset definition
    dataset = JVSDataset(melpath)
    collator = AudioCollate()

    # Model definition
    model = Model(dim=512)
    model.cuda()
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0005, betas=(0.9, 0.999))

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
            loss = reconstruction_loss(y, x)
            loss += kl_loss(sigma, mu)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if iteration % iterations == 0:
                torch.save(model.state_dict(), f"{modeldir}/model_{iteration}.pt")

            print(f"iteration: {iteration} loss: {loss.data}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Oneshot")
    parser.add_argument("--e", type=int, default=2000, help="the number of epochs")
    parser.add_argument("--b", type=int, default=32, help="batch size")
    parser.add_argument("--i", type=int, default=2000, help="interval")

    args = parser.parse_args()

    melpath = Path("./melspectrogram/train/")
    modeldir = Path("./modeldir")
    modeldir.mkdir(exist_ok=True)

    train(args.e, args.b, args.i, melpath, modeldir)

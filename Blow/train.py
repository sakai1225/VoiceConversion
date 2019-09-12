import numpy as np
import torch
import argparse
import torch.nn as nn
import torch.nn.functional as F

from pathlib import Path
from torch.utils.data import DataLoader
from model import Blow
from dataset import MultiSpeakerDataset, VoiceCollate
from tqdm import tqdm


def gaussian_log_p(z):
    log = 0.5 * np.log(2 * np.pi).item()

    return - log - 0.5 * (z ** 2)


def blow_loss(z, logdet):
    _, size = z.size()
    log_p = gaussian_log_p(z).sum(1)
    nll = - log_p - logdet
    nll /= size

    return nll.mean()


def train(epochs, batchsize, path, nblocks, nflows):
    # Dataset definition
    dataset = MultiSpeakerDataset(path)
    print(dataset)
    collator = VoiceCollate()

    # Model Definition & Optimizer Definition
    blow = Blow(nblocks, nflows)
    blow.cuda()
    blow.train()
    optimizer = torch.optim.Adam(blow.parameters(), lr=0.0001)

    for epoch in range(epochs):
        dataloader = DataLoader(dataset,
                                batch_size=batchsize,
                                shuffle=True,
                                drop_last=True,
                                collate_fn=collator)

        for i, data in enumerate(tqdm(dataloader)):
            s, s_label, t, t_label = data

            z, logdet = blow.forward(s, s_label)
            loss = blow_loss(z, logdet)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            print(f"iteration: {i} loss: {loss.data}")

        torch.save(blow.state_dict(), f"blow.model")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Blow")
    parser.add_argument("--e", type=int, default=1000, help="the number of epochs")
    parser.add_argument("--b", type=int, default=32, help="batch size")
    parser.add_argument("--nb", type=int, default=8, help="the number of blocks")
    parser.add_argument("--nf", type=int, default=12, help="the number of flows")

    args = parser.parse_args()

    data_path = Path("./starganvc/")

    train(args.e, args.b, data_path, args.nb, args.nf)
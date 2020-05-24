
# -*- coding: utf-8 -*-
import torch
import argparse
import tensorboardX as tbx
import torch.nn as nn
import numpy as  np

from model import Generator, Discriminator
from dataset import AudioCollator, AudioDataset
from pathlib import Path
from tqdm import tqdm
from torch.utils.data import DataLoader
from librosa.output import write_wav

def test(save_epoch, batchsize, data_path, save_path, modeldir, cls_num):

    # Dataset definition
    dataset = AudioDataset(data_path)
    collator = AudioCollator(cls_num)

    # Model & Optimizer definition
    generator = Generator(cls_num=cls_num)
    generator.load_state_dict(torch.load(f"{modeldir}/generator_{save_epoch - 1}.model"))
    generator.eval()                    # evaluation mode

    # Data loader
    dataloader = DataLoader(dataset,
                            batch_size=batchsize,
                            shuffle=False,
                            collate_fn=collator,
                            drop_last=True)
    dataloader = tqdm(dataloader)
    output = []

    # Evaluation
    for i, data in enumerate(dataloader):
        x_sp, x_label, y_label = data
        x_to_y = torch.cat([y_label, x_label], dim=1)
        y_to_x = torch.cat([x_label, y_label], dim=1)
        x_to_x = torch.cat([x_label, x_label], dim=1)

        # Generator update
        y_eval = generator(x_sp, x_to_y)
        y_npy  = y_eval.to('cpu').detach().numpy().flatten()

        # Save to List
        output.append(y_npy)

    # Writer
    path = str(Path(save_path))+'.wav'
    write_wav(path, np.array(output), sr=22050)
    #write_wav(save_path +'output.wav', np.array(output), sr=22050)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="StarGANVC2-pytorch")
    parser.add_argument("--e", type=int, default=1000, help="saved epoch number")
    parser.add_argument("--b", type=int, default=8, help="batch size")
    parser.add_argument("--n", type=int, default=4, help="the number of classes")

    args = parser.parse_args()

    data_path = Path("./StarGAN-VC2/data/speakers_test/")
    save_path = Path("./StarGAN-VC2/conversion/output")
    save_path.mkdir(exist_ok=True)
    modeldir = Path("./StarGAN-VC2/modeldir")
    #data_path = Path("./StarGAN-VC2/dataset")

    test(args.e, args.b, data_path, save_path, modeldir, args.n)
    #test(args.e, args.b, data_path, save_path, modeldir, args.n, args.i)
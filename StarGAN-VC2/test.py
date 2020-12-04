# -*- coding: utf-8 -*-
import torch
import argparse
import torch.nn as nn
import numpy as  np

from model import Generator, Discriminator
from dataset import AudioCollator, AudioDataset_test
from pathlib import Path
from tqdm import tqdm
from torch.utils.data import DataLoader
# from librosa.output import write_wav
from synthesizer import synthesized, file_name, dir_name


Classes = ["SF1", "SF2", "TM1", "TM2"]

def test(save_epoch, st, data_path, save_path, modeldir):

    # Dataset definition
    dataset = AudioDataset_test(data_path, Classes)
    file_list = dataset.set_speakers(st[0], st[1])
    collator = AudioCollator(4)

    # Model & Optimizer definition
    generator = Generator(cls_num=4)
    generator.load_state_dict(torch.load(f"{modeldir}/generator_{save_epoch}.model"))
    generator.cuda()
    generator.eval()  # evaluation mode

    save_path_sp = save_path/Path('calc_sp')
    save_path_sp.mkdir(exist_ok=True)

    # Evaluation
    for i, (data, file) in enumerate(zip(dataset, file_list)):

        #   Set Data
        x_label, y_label, x_sp = data
        n = len(x_sp)
        b_n = -(- n // 128)

        x_sp.resize((b_n, 1, 128, x_sp.shape[-1]), refcheck = False)
        x_label = np.repeat(collator._make_onehot(x_label)[np.newaxis,:], b_n, axis=0)
        y_label = np.repeat(collator._make_onehot(y_label)[np.newaxis,:], b_n, axis=0)

        x_sp = collator._totensor(x_sp)
        x_label = collator._totensor(x_label)
        y_label = collator._totensor(y_label)
        x_to_y = torch.cat([y_label, x_label], dim=1)

        #   Conversion
        y_eval = generator(x_sp, x_to_y)
        y_npy = y_eval.to('cpu').detach().numpy()
        y_npy = y_npy.reshape((-1, y_npy.shape[-1]))[:n, :]

        #   Save to npy
        name = file_name(file)
        np.save(str(save_path_sp) + '/' + name + '.npy', y_npy)

    #   Create synthesized wav
    name = dir_name(dir_name(file_list[0]))
    synthesized(name + '/calc_f0/',
                name + '/calc_ap/',
                str(save_path_sp)+'/',
                str(save_path)+'/')

    print('Finish')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="StarGANVC2-pytorch")
    parser.add_argument("--epoch", type=int, default=30000, help="saved epoch number")
    parser.add_argument("--batchsize", type=int, default=8, help="batch size")

    args = parser.parse_args()

    # Set of conversion object: (source, target)
    # 0: "SF1", 1: "SF2", 2: "TM1", 3: "TM2"
    convert_from    = 0
    convert_to      = 0
    name            = Classes[convert_from]+'_to_'+Classes[convert_to]
    conversion_set = (convert_from, convert_to)

    data_path = Path('./dataset_test/')
    save_path = Path('./dataset_test/'+name+'/30000/')
    save_path.mkdir(exist_ok=True)
    modeldir = Path("E:/modeldir")

    # Notice: batchsize must be 1
    test(args.epoch, conversion_set, data_path, save_path, modeldir)

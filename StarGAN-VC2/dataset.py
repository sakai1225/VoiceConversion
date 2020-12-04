import pyworld as pw
import numpy as np
import copy
import torch
import random

from librosa.core import load
from pathlib import Path
from torch.utils.data import Dataset


class AudioPreprocess:
    def __init__(self, path, dirlist, outdir, sr=16000):  # sr=22050
        self.path = path
        self.dirlist = dirlist
        self.outdir = outdir
        self.sr = sr

    def _load(self, path):
        audio, _ = load(path, sr=self.sr)

        return audio

    def audio2accousticfeatures(self, audio):
        """Conversion audio to accoustic features

        Args:
            audio (numpy.float): audio data
            sr (int, optional): Defaults to 16000. sampling rate

        Returns:
            numpy.float: f0, fundamental frequency
            numpy.float: sp, spectral envelope
            numpy.float: aperiodicity
        """

        audio = audio.astype(np.float)
        f0, t = pw.dio(audio, self.sr)
        f0 = pw.stonemask(audio, f0, t, self.sr)
        sp = pw.cheaptrick(audio, f0, t, self.sr)
        ap = pw.d4c(audio, f0, t, self.sr)

        return f0, sp, ap

    def encode_sp(self, sp, mel_bins=36):
        """Conversion spectral envelope to mel spectral envelope

        Args:
            sp (numpy.float): Spectral envelope
            sr (int, optional): Defaults to 16000. Sampling rate
            mel_bins (int, optional): Defaults to 36. The number of mel bins

        Returns:
            numpy.float: mel spectral envelope
        """

        encoded_sp = pw.code_spectral_envelope(sp, self.sr, mel_bins)

        return encoded_sp

    def __call__(self):
        for index, directory in enumerate(self.dirlist):

            #   フォルダ作成
            outdir = self.outdir / Path(directory)
            outdir_f0 = outdir / Path('calc_f0')
            outdir_ap = outdir / Path('calc_ap')
            outdir_sp = outdir / Path('calc_sp')
            outdir_f0.mkdir(exist_ok=True)
            outdir_ap.mkdir(exist_ok=True)
            outdir_sp.mkdir(exist_ok=True)

            indir = self.path / Path(directory)
            indir_list = list(indir.glob("*.wav"))

            for inpath in indir_list:
                audio = self._load(str(inpath))
                f0, sp, ap = self.audio2accousticfeatures(audio)
                sp = self.encode_sp(sp)
                name = Path(str(inpath.name[:-4]+ '.npy'))
                outpath_f0 = outdir_f0 / name
                outpath_sp = outdir_sp / name
                outpath_ap = outdir_ap / name
                np.save(str(outpath_f0), f0)
                np.save(str(outpath_sp), sp)
                np.save(str(outpath_ap), ap)


def decode_sp(sp, sr=16000, fftsize=1024):
    """Conversion mel spectral envelope to spectral envelope
    Args:
        sp (numpy.float): Mel spectral envelope
        sr (int, optional): Defaults to 16000. Sampling rate
        fftsize (int, optional): Defaults to 1024. The size of fft

    Returns:
        numpy.float: mel spectral envelope
    """
    encoded_sp = pw.decode_spectral_envelope(sp.astype(np.double), sr, fftsize)

    return encoded_sp


class AudioDataset(Dataset):
    def __init__(self, path: Path, dirlist):
        self.path = path
        self.pathlist = list(path.glob("*/calc_sp/*.npy"))
        self.dirlist = dirlist      # List of speakers (4)

    def __len__(self):
        return len(self.pathlist)

    def _label_remove(self, copy_list, label):
        copy_list.remove(str(label))

        return copy_list

    def __getitem__(self, idx):
        copy_list = copy.copy(self.dirlist)
        [src_ind, tgt_ind] = random.sample(range(len(copy_list)), 2)    # Pick up 2 speakers from 4 speakers
        src_rnd = copy_list[src_ind]            # Picked up list

        src_path = self.path / Path(src_rnd)    # path/speaker_list (ex:dataset/SF1)
        src_list = list(src_path.glob("calc_sp/*.npy")) # numpy list (ex:dataset/SF1/*.npy)
        src_name = np.random.choice(src_list)
        src_sp = np.load(src_name)

        return src_ind, tgt_ind, src_sp


class AudioDataset_test(AudioDataset):

    def __init__(self, path: Path, dirlist):
        super().__init__(path, dirlist)
        self.path = path
        self.dirlist = dirlist      # List of speakers (4)
        self.src_list = []

    def __len__(self):
        return len(self.src_list)

    def set_speakers(self, src_ind, tgt_ind):
        self.src_ind = src_ind
        self.tgt_ind = tgt_ind
        copy_list = copy.copy(self.dirlist)
        src_name = copy_list[src_ind]
        src_path = self.path / Path(src_name)  # path/speaker_list (ex:dataset/SF1)
        self.src_list = list(src_path.glob('calc_sp/*.npy'))  # numpy list (ex:dataset/SF1/*.npy)
        return self.src_list

    def __getitem__(self, idx):
        src_name = self.src_list[idx]
        src_sp = np.load(src_name)
        return self.src_ind, self.tgt_ind, src_sp


class AudioCollator:
    def __init__(self, cls_num):
        self.cls_num = cls_num

    def _make_onehot(self, label):
        onehot = np.zeros(self.cls_num)
        onehot[int(label)] = 1

        return onehot

    @staticmethod
    def _normalize(x, epsilon=1e-8):
        x_mean = np.mean(x, axis=1, keepdims=True)
        x_std = np.std(x, axis=1, keepdims=True)

        return (x - x_mean) / x_std

    @staticmethod
    def _crop(sp, upper_bound=128):
        if sp.shape[0] < upper_bound + 1:
            sp = np.pad(sp, ((0, upper_bound - sp.shape[0] + 2), (0, 0)), 'constant', constant_values=0)

        start_point = np.random.randint(sp.shape[0] - upper_bound)
        cropped = sp[start_point: start_point + upper_bound, :]

        return cropped

    @staticmethod
    def _totensor(array_list):
        array = np.array(array_list).astype(np.float32)
        array = torch.cuda.FloatTensor(array)

        return array

    def __call__(self, batch):
        x_label = []
        x_sp = []
        y_label = []

        for b in batch:
            x, y, sp = b

            x = self._make_onehot(x)
            y = self._make_onehot(y)

            sp = self._normalize(sp)
            sp = self._crop(sp)
            sp = sp[np.newaxis, :, :]

            x_label.append(x)
            y_label.append(y)
            x_sp.append(sp)

        x_label = self._totensor(x_label)
        y_label = self._totensor(y_label)
        x_sp = self._totensor(x_sp)

        return (x_sp, x_label, y_label)

class AudioCollator_test(AudioCollator):
    def __init__(self, cls_num):
        super().__init__(cls_num)
        self.cls_num = cls_num

    def _make_onehot(self, label):
        onehot = np.zeros(self.cls_num)
        onehot[int(label)] = 1

        return onehot

    @staticmethod
    def _normalize(x, epsilon=1e-8):
        x_mean = np.mean(x, axis=1, keepdims=True)
        x_std = np.std(x, axis=1, keepdims=True)

        return (x - x_mean) / x_std

    @staticmethod
    def _crop(sp, upper_bound=128):
        if sp.shape[0] < upper_bound + 1:
            sp = np.pad(sp, ((0, upper_bound - sp.shape[0] + 2), (0, 0)), 'constant', constant_values=0)

        start_point = np.random.randint(sp.shape[0] - upper_bound)
        cropped = sp[start_point: start_point + upper_bound, :]

        return cropped

    @staticmethod
    def _totensor(array_list):
        array = np.array(array_list).astype(np.float32)
        array = torch.cuda.FloatTensor(array)

        return array

    def __call__(self, batch):
        x_label = []
        x_sp = []
        y_label = []

        for b in batch:
            x, y, sp = b

            x = self._make_onehot(x)
            y = self._make_onehot(y)

            sp = self._normalize(sp)
            sp = self._crop(sp)
            sp = sp[np.newaxis, :, :]

            x_label.append(x)
            y_label.append(y)
            x_sp.append(sp)

        x_label = self._totensor(x_label)
        y_label = self._totensor(y_label)
        x_sp = self._totensor(x_sp)

        return (x_sp, x_label, y_label)


if __name__ == "__main__":
    path_train = Path("./data/speakers/")  # path = Path("./Dataset/Speech/")
    path_test = Path("./data/speakers_test/")
    dirlist = ["SF1", "SF2", "TM1", "TM2"]

    outdir_train = Path("./dataset/")
    outdir_test = Path("./dataset_test/")
    outdir_train.mkdir(exist_ok=True)
    outdir_test.mkdir(exist_ok=True)

    preprocess_train = AudioPreprocess(path_train, dirlist, outdir_train)
    preprocess_test = AudioPreprocess(path_test, dirlist, outdir_test)

    preprocess_train()
    preprocess_test()

import torch
import librosa
import argparse
import os
import numpy as np

from torch.utils.data import Dataset
from librosa.core import load
from pathlib import Path
from utils import get_spectrograms
from hyperparams import Hyperparams as hp


class JVSPreprocess:
    def __init__(self, path: Path, mel_path, split=80, sr=22050):
        self.path = path
        self.melpath = mel_path
        self.sr = sr
        self.dirlist = [dirs for dirs in os.listdir(str(path)) if not os.path.isfile(str(path) + dirs)]
        print(self.dirlist)

    @staticmethod
    def get_spectrograms(fpath):
        '''Returns normalized log(melspectrogram) and log(magnitude) from `sound_file`.
        Args:
        sound_file: A string. The full path of a sound file.
        Returns:
        mel: A 2d array of shape (T, n_mels) <- Transposed
        mag: A 2d array of shape (T, 1+n_fft/2) <- Transposed
        '''

        # Loading sound file
        y, sr = librosa.load(fpath, sr=hp.sr)

        # Trimming
        y, _ = librosa.effects.trim(y, top_db=hp.top_db)

        # Preemphasis
        y = np.append(y[0], y[1:] - hp.preemphasis * y[:-1])

        # stft
        linear = librosa.stft(y=y,
                              n_fft=hp.n_fft,
                              hop_length=hp.hop_length,
                              win_length=hp.win_length)

        # magnitude spectrogram
        mag = np.abs(linear)  # (1+n_fft//2, T)

        # mel spectrogram
        mel_basis = librosa.filters.mel(hp.sr, hp.n_fft, hp.n_mels)  # (n_mels, 1+n_fft//2)
        mel = np.dot(mel_basis, mag)  # (n_mels, t)

        # to decibel
        mel = 20 * np.log10(np.maximum(1e-5, mel))
        mag = 20 * np.log10(np.maximum(1e-5, mag))

        # normalize
        mel = np.clip((mel - hp.ref_db + hp.max_db) / hp.max_db, 1e-8, 1)
        mag = np.clip((mag - hp.ref_db + hp.max_db) / hp.max_db, 1e-8, 1)

        # Transpose
        mel = mel.T.astype(np.float32)  # (T, n_mels)
        mag = mag.T.astype(np.float32)  # (T, 1+n_fft//2)

        return mel, mag

    def _convert(self, path):
        melsp, _ = self.get_spectrograms(path)

        return melsp.T

    def _save(self, speaker_path, speaker_out_path):
        nonpara_path = f"{str(speaker_path)}/nonpara30/wav24kHz16bit/"
        nonpara_list = os.listdir(nonpara_path)
        for wav in nonpara_list:
            nonpara_wav = nonpara_path + wav
            melsp = self._convert(nonpara_wav)
            np.save(f"{str(speaker_out_path)}/{wav}", melsp)

        para_path = f"{str(speaker_path)}/parallel100/wav24kHz16bit/"
        para_list = os.listdir(para_path)
        for wav in para_list:
            para_wav = para_path + wav
            melsp = self._convert(para_wav)
            np.save(f"{str(speaker_out_path)}/{wav}", melsp)

    def __call__(self):
        for train_speaker in self.dirlist:
            speaker_path = self.path/Path(train_speaker)
            print(speaker_path)
            speaker_out_path = self.melpath/Path(train_speaker)
            speaker_out_path.mkdir(exist_ok=True)
            self._save(speaker_path, speaker_out_path)


class JVSDataset(Dataset):
    def __init__(self,
                 mel_path: Path,
                 extension=".npy"):

        self.melpath = mel_path
        self.mellist = list(mel_path.glob(f"**/*{extension}"))

    def __len__(self):
        return len(self.mellist)

    def __getitem__(self, idx):
        melpath = self.mellist[idx]

        return melpath


class AudioCollate:
    def __init__(self, time_width=128):
        self.tw = time_width

    @staticmethod
    def _normalize(audio):
        audio = librosa.power_to_db(audio, ref=np.max)
        audio = (audio + 80) / 80

        return audio

    @staticmethod
    def _crop(melsp, upper_bound):
        if melsp.shape[1] < upper_bound + 1:
            melsp = np.pad(melsp,
                           ((0, 0), (0, upper_bound - melsp.shape[1] + 2)),
                           "constant",
                           constant_values=0)

        start_point = np.random.randint(melsp.shape[1] - upper_bound)
        cropped = melsp[:, start_point: start_point + upper_bound]

        return cropped

    def _prepare(self, melpath):
        melsp = np.load(melpath)
        melsp = self._crop(melsp, self.tw)

        return melsp

    @staticmethod
    def _totensor(array_list):
        array = np.array(array_list).astype(np.float32)
        tensor = torch.cuda.FloatTensor(array)

        return tensor

    def __call__(self, batch):
        melsp_box = []
        for melpath in batch:
            melsp = self._prepare(melpath)
            melsp_box.append(melsp)

        melsp = self._totensor(melsp_box)

        return melsp


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Oneshot Preprocess")
    parser.add_argument("--jvs_path", type=Path, help="jvs path")
    parser.add_argument("--mel_path", type=Path, help="preprocess output path")
    args = parser.parse_args()

    mel_path = args.mel_path
    mel_path.mkdir(exist_ok=True)

    preprocess = JVSPreprocess(args.jvs_path, mel_path)
    preprocess()

import torch
import librosa
import numpy as np
import os

from torch.utils.data import Dataset
from librosa.core import load
from pathlib import Path


class JVSPreprocess:
    def __init__(self, path: Path, mel_path, split=80, sr=22050):
        self.path = path
        self.melpath = mel_path
        self.sr = sr
        self.dirlist = [dirs for dirs in os.listdir(str(path)) if not os.path.isfile(str(path) + dirs)]
        print(self.dirlist)

    def audio2melsp(self,
                    audio,
                    melsize=512,
                    fftsize=2048,
                    window_size=1100,
                    window_shift_size=276):
        """Conversion audio to mel-spectrogram
        
        Args:
            audio (numpy.float): audio data
            melsize (int, optional): Defaults to 80. the number of mel bins
            fftsize (int, optional): Defaults to 512. FFT size
            window_size (int, optional): Defaults to 400. The width of window
            window_shift_size (int, optional): Defaults to 160. The shift size of window
        
        Returns:
            numpy.float: mel-spectrogram
        """

        window = librosa.filters.get_window('hann', window_size, fftbins=True)
        window = window.reshape((-1, 1))

        frame = librosa.util.frame(audio, frame_length=window_size, hop_length=window_shift_size)
        frame = window * frame

        stft = np.fft.rfft(frame.T, n=fftsize)
        mel = librosa.feature.melspectrogram(S=np.abs(stft.T), n_mels=melsize)

        return mel

    def _convert(self, path):
        audio, _ = load(path, sr=self.sr, mono=True)
        melsp = self.audio2melsp(audio)

        return melsp

    def _save(self, speaker_path, speaker_out_path):
        nonpara_path = f"{str(speaker_path)}/nonpara30/wav24kHz16bit/"
        nonpara_list = os.listdir(nonpara_path)
        for wav in nonpara_list:
            nonpara_wav = nonpara_path + wav
            melsp = self._convert(nonpara_wav)
            print(melsp.shape)
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
            speaker_out_path = self.melpath/Path(train_speaker)
            speaker_out_path.mkdir(exist_ok=True)
            self._save(speaker_path, speaker_out_path)


class JVSDataset(Dataset):
    def __init__(self, mel_path: Path):
        self.melpath = mel_path
        self.mellist = list(mel_path.glob("**/*.npy"))

    def __len__(self):
        return len(self.mellist)

    def __getitem__(self, idx):
        melpath = self.mellist[idx]

        return melpath


class AudioCollate:
    def __init__(self):
        pass

    @staticmethod
    def _normalize(audio):
        audio = librosa.power_to_db(audio, ref=np.max)
        audio = (audio + 80) / 80

        return audio

    @staticmethod
    def _crop(melsp, upper_bound=128):
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
        melsp = self._normalize(melsp)
        melsp = self._crop(melsp)

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
    jvs_path = Path("./jvs_ver1/")
    mel_path = Path("./melspectrogram/")
    mel_path.mkdir(exist_ok=True)
    preprocess = JVSPreprocess(jvs_path, mel_path)
    preprocess()

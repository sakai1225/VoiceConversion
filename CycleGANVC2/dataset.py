import chainer
import copy
import numpy as np
import pyworld as pw

from pathlib import Path
from chainer import cuda

xp = cuda.cupy
cuda.get_device(0).use()


class DatasetLoader:
    def __init__(self,
                 src_path: Path,
                 tgt_path: Path,
                 extension=".npy",
                 time_width=128,
                 mel_bins=36,
                 sampling_rate=22050
                 ):

        self.src_path = src_path
        self.tgt_path = tgt_path

        self.extension = extension
        self.time_width = time_width
        self.mel_bins = mel_bins
        self.sr = sampling_rate

        self.srclist, self.srclen = self._get_list_and_length(self.src_path)
        self.tgtlist, self.tgtlen = self._get_list_and_length(self.tgt_path)

    def __repr__(self):
        return f"source length: {self.srclen} target length {self.tgtlen}"

    def _get_list_and_length(self, path: Path):
        pathlist = list(path.glob(f"*{self.extension}"))
        pathlen = len(pathlist)

        return pathlist, pathlen

    @staticmethod
    def _encode_sp(sp, sr, mel_bins):
        """Conversion spectral envelope to mel spectral envelope
        
        Args:
            sp (numpy.float): Spectral envelope
            sr (int, optional): Defaults to 16000. Sampling rate
            mel_bins (int, optional): Defaults to 36. The number of mel bins
        
        Returns:
            numpy.float: mel spectral envelope
        """

        encoded_sp = pw.code_spectral_envelope(sp, sr, mel_bins)

        return encoded_sp

    @staticmethod
    def _normalize(x, epsilon=1e-8):
        x_mean = np.mean(x, axis=1, keepdims=True)
        x_std = np.std(x, axis=1, keepdims=True)

        return (x - x_mean) / x_std

    @staticmethod
    def _crop(sp, upper_bound):
        if sp.shape[0] < upper_bound + 1:
            sp = np.pad(sp, ((0, upper_bound-sp.shape[0] + 2), (0, 0)), 'constant', constant_values=0)

        start_point = np.random.randint(sp.shape[0] - upper_bound)
        cropped = sp[start_point: start_point + upper_bound, :]

        return cropped

    @staticmethod
    def _variable(array_list, array_type='float'):
        if array_type == 'float':
            return chainer.as_variable(xp.array(array_list).astype(xp.float32))

        else:
            return chainer.as_variable(xp.array(array_list).astype(xp.int32))

    def _prepare_sp(self, path):
        sp = np.load(str(path))
        sp = self._encode_sp(sp, sr=self.sr, mel_bins=self.mel_bins)
        sp = self._normalize(sp)
        sp = self._crop(sp, upper_bound=self.time_width)

        return sp

    def train(self, batchsize):
        x_sp_box = []
        y_sp_box = []

        for _ in range(batchsize):
            path = np.random.choice(self.srclist)
            sp = self._prepare_sp(path)

            x_sp_box.append(sp[np.newaxis, :])

            path = np.random.choice(self.tgtlist)
            sp = self._prepare_sp(path)

            y_sp_box.append(sp[np.newaxis, :])

        x = self._variable(x_sp_box)
        y = self._variable(y_sp_box)

        return x, y

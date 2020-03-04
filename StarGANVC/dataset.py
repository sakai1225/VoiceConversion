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
                 data_path: Path,
                 extension=".npy",
                 time_width=128,
                 mel_bins=36,
                 sampling_rate=22050
                 ):

        self.data_path = data_path
        self.belong_list = [path.name for path in self.data_path.iterdir()]
        self.belong_hashmap = self._hashmap(self.belong_list)
        self.number = len(self.belong_list)

        self.extension = extension
        self.time_width = time_width
        self.mel_bins = mel_bins
        self.sr = sampling_rate

    @staticmethod
    def _encode_sp(sp, sr, mel_bins):
        """
        Conversion spectral envelope to mel spectral envelope
        
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
    def _hashmap(belong_list):
        belong_len = len(belong_list)
        hashmap = {}
        for belong, index in zip(belong_list, range(belong_len)):
            hashmap[belong] = index

        return hashmap

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
    def _label_remove(label_list, source):
        label_list.remove(source)

        return label_list

    @staticmethod
    def _variable(array_list, array_type='float'):
        if array_type == 'float':
            return chainer.as_variable(xp.array(array_list).astype(xp.float32))

        else:
            return chainer.as_variable(xp.array(array_list).astype(xp.int32))

    def _onehot_convert(self, label):
        onehot = np.zeros(self.number)
        onehot[self.belong_hashmap[label]] = 1

        return onehot

    def _prepare_sp(self, path):
        sp = np.load(str(path))
        sp = self._encode_sp(sp, sr=self.sr, mel_bins=self.mel_bins)
        sp = self._normalize(sp)
        sp = self._crop(sp, upper_bound=self.time_width)

        return sp

    def _get_path_onehot(self, label_list):
        rnd_belong = np.random.choice(label_list)
        pathlist = list((self.data_path / Path(str(rnd_belong))).glob(f"*{self.extension}"))

        sp_path = np.random.choice(pathlist)
        onehot = self._onehot_convert(rnd_belong)
        sp = self._prepare_sp(sp_path)

        return rnd_belong, sp, onehot

    def train(self, batchsize):
        x_sp_box = []
        x_label_box = []
        y_sp_box = []
        y_label_box = []
        for _ in range(batchsize):
            label_list = copy.copy(self.belong_list)
            rnd_belong, sp, onehot = self._get_path_onehot(label_list)

            x_sp_box.append(sp[np.newaxis, :])
            x_label_box.append(onehot)

            label_list = self._label_remove(label_list, rnd_belong)
            _, sp, onehot = self._get_path_onehot(label_list)

            y_sp_box.append(sp[np.newaxis, :])
            y_label_box.append(onehot)

        x_sp = self._variable(x_sp_box)
        x_label = self._variable(x_label_box, array_type='float')
        y_sp = self._variable(y_sp_box)
        y_label = self._variable(y_label_box, array_type='float')

        return (x_sp, x_label, y_sp, y_label)

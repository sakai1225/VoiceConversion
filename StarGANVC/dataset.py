import chainer
import numpy as np
import copy

from pathlib import Path
from chainer import cuda
from lain.audio.layer.converter import encode_sp

xp = cuda.cupy
cuda.get_device(0).use()


class DatasetLoader:
    def __init__(self, data_path):
        self.data_path = data_path
        self.label_list = [0, 1, 2, 3]

    @staticmethod
    def _normalize(x, epsilon=1e-8):
        x_mean = np.mean(x, axis=1, keepdims=True)
        x_std = np.std(x, axis=1, keepdims=True)

        return (x - x_mean) / x_std

    @staticmethod
    def _crop(sp, upper_bound=128):
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

    @staticmethod
    def _onehot_convert(label):
        onehot = np.zeros(4)
        onehot[label] = 1

        return onehot

    def _prepare_sp(self, path):
        sp = np.load(str(path))
        sp = self._normalize(encode_sp(sp, sr=22050, mel_bins=36))
        sp = self._crop(sp, upper_bound=128)

        return sp

    def train(self, batchsize):
        x_sp_box = []
        x_label_box = []
        y_sp_box = []
        y_label_box = []
        for _ in range(batchsize):
            label_list = copy.copy(self.label_list)
            rnd = np.random.choice(label_list)
            onehot = self._onehot_convert(rnd)
            pathlist = list((self.data_path/Path(str(rnd))).glob('*.npy'))
            sp_path = np.random.choice(pathlist)
            sp = self._prepare_sp(sp_path)

            x_sp_box.append(sp[np.newaxis, :])
            x_label_box.append(onehot)

            label_list = self._label_remove(label_list, rnd)
            rnd = np.random.choice(label_list)
            onehot = self._onehot_convert(rnd)
            pathlist = list((self.data_path/Path(str(rnd))).glob('*.npy'))
            sp_path = np.random.choice(pathlist)
            sp = self._prepare_sp(sp_path)

            y_sp_box.append(sp[np.newaxis, :])
            y_label_box.append(onehot)

        x_sp = self._variable(x_sp_box)
        x_label = self._variable(x_label_box, array_type='float')
        y_sp = self._variable(y_sp_box)
        y_label = self._variable(y_label_box, array_type='float')

        return (x_sp, x_label, y_sp, y_label)
import chainer
import librosa
import numpy as np

from pathlib import Path
from chainer import cuda
from librosa.core import load
from lain.audio.layer.converter import audio2melsp

xp = cuda.cupy
cuda.get_device(0).use()


class DatasetLoader:
    def __init__(self, src_path: Path, tgt_path: Path):
        self.src_path = src_path
        self.tgt_path = tgt_path

        self.src_list = list(self.src_path.glob("*.wav"))
        self.tgt_list = list(self.tgt_path.glob("**/**/*.wav"))

    def __repr__(self):
        return f"Source: {len(self.src_list)} Target: {len(self.tgt_list)}"

    @staticmethod
    def _normalize(x, epsilon=1e-8):
        x = librosa.power_to_db(x, ref=np.max)
        x = (x + 80) / 80

        return x

    @staticmethod
    def _crop(sp, upper_bound=128):
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
        sp, _ = load(str(path), sr=22050)
        sp = self._normalize(audio2melsp(sp, melsize=80).transpose(1, 0))
        sp = self._crop(sp, upper_bound=128)

        return sp

    def train(self, batchsize):
        x_sp_box = []
        y_sp_box = []

        for _ in range(batchsize):
            sp_path = np.random.choice(self.src_list)
            sp = self._prepare_sp(sp_path)

            x_sp_box.append(sp[np.newaxis, :])

            sp_path = np.random.choice(self.tgt_list)
            sp = self._prepare_sp(sp_path)

            y_sp_box.append(sp[np.newaxis, :])

        x_sp = self._variable(x_sp_box)
        y_sp = self._variable(y_sp_box)

        return (x_sp, y_sp)

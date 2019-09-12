import numpy as np
import copy
import torch

from pathlib import Path
from torch.utils.data import Dataset
from librosa.core import load


class MultiSpeakerDataset(Dataset):
    def __init__(self, path: Path):
        self.sumlist = list(path.glob('**/*.wav'))
        self.path = path
        self.label = [0, 1, 2, 3]

    @staticmethod
    def _label_remove(label, source):
        label.remove(source)

        return label

    def __str__(self):
        return f"The number of data: {len(self.sumlist)}"

    def __len__(self):
        return len(self.sumlist)

    def __getitem__(self, idx):
        label = copy.copy(self.label)
        rnd_s = np.random.choice(label)
        pathlist = list((self.path/Path(str(rnd_s))).glob('*.wav'))
        s_path = np.random.choice(pathlist)

        label = self._label_remove(label, rnd_s)
        rnd_t = np.random.choice(label)
        pathlist = list((self.path/Path(str(rnd_t))).glob('*.wav'))
        t_path = np.random.choice(pathlist)

        return (s_path, rnd_s, t_path, rnd_t)


class VoiceCollate:
    def __init__(self, sr=22050):
        self.sr = sr

    def _random_crop(self, audio, size=4096):
        length = audio.shape[0]
        start_point = np.random.randint(length - size)
        cropped_audio = audio[start_point: start_point + size]

        return cropped_audio

    def _normalize(self, audio):
        mean = np.mean(audio)
        audio = (audio - mean) / (np.max(np.abs(audio)))

        return audio

    def _prepare(self, apath):
        audio, _ = load(str(apath), self.sr)
        audio = self._normalize(audio)
        audio = audio.astype(np.float32)
        audio = self._random_crop(audio)

        return audio

    def _totensor(self, array_list, array_type="float"):
        array = np.array(array_list)

        if array_type == "float":
            ten = torch.Tensor(array)
        else:
            ten = torch.LongTensor(array)

        ten = ten.cuda()

        return ten

    def __call__(self, batch):
        s_box, s_label_box, t_box, t_label_box = [], [], [], []
        for data in batch:
            s_path, rnd_s, t_path, rnd_t = data
            s_audio = self._prepare(s_path)
            t_audio = self._prepare(t_path)

            s_box.append(s_audio)
            s_label_box.append(rnd_s)
            t_box.append(t_audio)
            t_label_box.append(rnd_t)

        s = self._totensor(s_box)
        s_label = self._totensor(s_label_box, array_type="long")
        t = self._totensor(t_box)
        t_label = self._totensor(t_label_box, array_type="long")

        return (s, s_label, t, t_label)

import pyworld as pw
import numpy as np

from scipy.io import wavfile

WAV_FILE = './StarGAN-VC2/data/speakers_test/SF1/200005.wav'

fs, data = wavfile.read(WAV_FILE)
data = data.astype(np.float)  # WORLDはfloat前提のコードになっているのでfloat型にしておく

_f0, t = pw.dio(data, fs)  # 基本周波数の抽出
f0 = pw.stonemask(data, _f0, t, fs)  # 基本周波数の修正
ap = pw.d4c(data, f0, t, fs)  # 非周期性指標の抽出

np.save('./StarGAN-VC2/calc_f0/f0.npy',f0)
np.save('./StarGAN-VC2/calc_ap/ap.npy', ap)

#sp = pw.cheaptrick(data, f0, t, fs)  # スペクトル包絡の抽出


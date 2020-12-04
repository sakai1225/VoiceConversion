# Vocoder using WORLD

import pyworld as pw
import numpy as np
import glob
import librosa
import os

from scipy.io import wavfile
from scipy.io.wavfile import write

def file_name(pathname):
    return os.path.splitext(os.path.basename(pathname))[0]


files = glob.glob('./StarGAN-VC2/data/speakers_test/**/*.wav')

for file in files:
    fs, data = wavfile.read(file)
    name = file_name(file)
    data = data.astype(np.float)  # WORLDはfloat前提のコードになっているのでfloat型にしておく

    _f0, t = pw.dio(data, fs)  # 基本周波数の抽出
    f0 = pw.stonemask(data, _f0, t, fs)  # 基本周波数の修正
    ap = pw.d4c(data, f0, t, fs)  # 非周期性指標の抽出
    sp = pw.cheaptrick(data, f0, t, fs)  # スペクトル包絡の抽出

    np.save(str('./StarGAN-VC2/calc_f0/' + name + '.npy'), f0)
    np.save(str('./StarGAN-VC2/calc_ap/' + name + '.npy'), ap)
    np.save(str('./StarGAN-VC2/calc_sp/sp_' + name + '.npy'), sp)

    synthesized = pw.synthesize(f0, sp, ap, fs)
    synthesized = 0.75/np.max(np.abs(synthesized)) * synthesized  # Normalization

    librosa.output.write_wav(str('./StarGAN-VC2/synthesized/' + name + '.wav'), synthesized, fs)


f0 = np.load('./StarGAN-VC2/calc_f0/200004.npy')
ap = np.load('./StarGAN-VC2/calc_ap/200004.npy')
sp = np.load('./StarGAN-VC2/calc_sp/sp_200004.npy')

synthesized = pw.synthesize(f0, sp, ap, 16000)
synthesized = 0.75/np.max(np.abs(synthesized)) * synthesized  # Normalization
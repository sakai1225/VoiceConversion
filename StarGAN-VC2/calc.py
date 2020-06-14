import pyworld as pw
import numpy as np
import glob
import librosa
import os

from scipy.io import wavfile
from scipy.io.wavfile import write

def file_name(path):
    return os.path.splitext(os.path.basename(filepath))[0]


files = glob.glob('./StarGAN-VC2/data/speakers_test/**/*.wav')

for file in files:
    fs, data = wavfile.read(file)
    name = file_bame(file)
    data = data.astype(np.float)  # WORLDはfloat前提のコードになっているのでfloat型にしておく

    _f0, t = pw.dio(data, fs)  # 基本周波数の抽出
    f0 = pw.stonemask(data, _f0, t, fs)  # 基本周波数の修正
    ap = pw.d4c(data, f0, t, fs)  # 非周期性指標の抽出
    sp = pw.cheaptrick(data, f0, t, fs)  # スペクトル包絡の抽出

    np.save(str('./StarGAN-VC2/calc_f0/' + name + '.npy'), f0)
    np.save(str('./StarGAN-VC2/calc_ap/' + name + '.npy'), ap)
    np.save(str('./StarGAN-VC2/calc_sp/sp_' + name + '.npy'), sp)
    synthesized = pw.synthesize(f0, sp, ap, fs)
    synthesized = 0.75/np.max(np.abs(synthesize)) * synthesize # Normalization

    librosa.output.write_wav(str('./StarGAN-VC2/synthesized/' + name + '.wav'), synthesized, fs)




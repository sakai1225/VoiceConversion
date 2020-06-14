import pyworld as pw
import numpy as np
import glob
import librosa

from scipy.io import wavfile
from scipy.io.wavfile import write

files = glob.glob('./StarGAN-VC2/data/speakers_test/**/*.wav')
i = 1 #ファイル名用

for file in files:
    fs, data = wavfile.read(file)
    data = data.astype(np.float)  # WORLDはfloat前提のコードになっているのでfloat型にしておく

    _f0, t = pw.dio(data, fs)  # 基本周波数の抽出
    f0 = pw.stonemask(data, _f0, t, fs)  # 基本周波数の修正
    ap = pw.d4c(data, f0, t, fs)  # 非周期性指標の抽出
    sp = pw.cheaptrick(data, f0, t, fs)  # スペクトル包絡の抽出

    np.save(str('./StarGAN-VC2/calc_f0/f0_' + str(i) + '.npy'), f0)
    np.save(str('./StarGAN-VC2/calc_ap/ap_' + str(i) + '.npy'), ap)
    np.save(str('./StarGAN-VC2/calc_sp/sp_' + str(i) + '.npy'), sp)
    synthesized = pw.synthesize(f0, sp, ap, fs)

    librosa.output.write_wav(str('./StarGAN-VC2/synthesized/' + str(i) + '.wav'), synthesized, fs)
    #write(str('./StarGAN-VC2/synthesized/' + str(i) + '.wav'), fs, synthesized)
    i += 1



import pyworld as pw
import numpy as np
import glob
import librosa
import os
from pathlib import Path
from scipy.io import wavfile
from dataset import decode_sp

data_dir = 'dataset_test/'
f0_dir = '/calc_f0/'
ap_dir = '/calc_ap/'
sp_dir = '/calc_sp/'
syn_dir = '/world_synth/'

def file_name(pathname):
    return os.path.splitext(os.path.basename(pathname))[0]

def dir_name(pathname):
    return os.path.dirname(pathname)

#   Synthesizer of WORLD
def synthesized(f0_path, ap_path, sp_path, wav_path, fs=16000):

    f0_files = glob.glob(f0_path + '*.npy')
    ap_files = glob.glob(ap_path + '*.npy')
    sp_files = glob.glob(sp_path + '*.npy')

    for f0_file, ap_file, sp_file in zip(f0_files, ap_files, sp_files):
        name = file_name(f0_file)
        f0 = np.load(f0_file)
        ap = np.load(ap_file)
        sp = decode_sp(np.load(sp_file))
        syn = pw.synthesize(f0, sp, ap, fs)
        syn = 0.75 / np.max(np.abs(syn)) * syn  # Normalization
        librosa.output.write_wav(str(wav_path) + name + '.wav', syn, fs)

#
#   Create Audio Features Files
#
if __name__ == "__main__":

    dirs = glob.glob(data_dir + '**')

    for i, dir in enumerate(dirs):

        print('[process] Calculate audio features in No.{0}/{1} folder'.format(i+1, len(dirs)))
        Path(dir + syn_dir).mkdir(exist_ok=True)

        #  Create Synthesized Wave files
        synthesized(dir + f0_dir,
                    dir + ap_dir,
                    dir + sp_dir,
                    dir + syn_dir)

    print('[save] Output audio features to npy files')
    print('[save] Output synthesized wave files')
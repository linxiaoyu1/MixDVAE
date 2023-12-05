## MixDVAE
## Copyright Inria
## Year 2023
## Contact : xiaoyu.lin@inria.fr

## MixDVAE is free software: you can redistribute it and/or modify
## it under the terms of the GNU General Public License as published by
## the Free Software Foundation, either version 3 of the License, or
## (at your option) any later version.

## MixDVAE is distributed in the hope that it will be useful,
## but WITHOUT ANY WARRANTY; without even the implied warranty of
## MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
## GNU General Public License for more details.
##
## You should have received a copy of the GNU General Public License
## along with this program, MixDVAE.  If not, see <http://www.gnu.org/licenses/> and the LICENSE file.

# MixDVAE has code derived from 
# (1) ArTIST, https://github.com/fatemeh-slh/ArTIST.
# (2) DVAE, https://github.com/XiaoyuBIE1994/DVAE, distributed under MIT License 2020 INRIA.

import os
import numpy as np
import soundfile as sf
import librosa
import pickle
from torch.utils import data
import matplotlib.pyplot as plt
import librosa.display
from pesq import pesq

# Parameters
s1_path = './CBF_process/train'
s2_path = './data/WSJ0/wsj0_si_tr_s'
save_dir = './data/Mixture_WSJ0_CBF_SNR_100frames_train'
data_suffix = 'wav'
wlen_sec = 64e-3
hop_percent = 0.25
fs = 16000
zp_percent = 0
wlen = wlen_sec * fs
wlen = int(np.power(2, np.ceil(np.log2(wlen)))) # pwoer of 2
hop = int(hop_percent * wlen)
nfft = wlen + zp_percent * wlen
win = np.sin(np.arange(0.5, wlen+0.5) / wlen * np.pi)
max_num = 10000
true_num = 0
seq_len_stft = 100
snr = -10
seq_len_temporal = (seq_len_stft - 1) * hop
import pdb; pdb.set_trace()

def cal_rms(amp):
    return np.sqrt(np.mean(amp**2, axis=-1))

def cal_adjusted_rms(clean_rms, snr):
    a = float(snr) / 20
    noise_rms = clean_rms / (10 ** a)
    return noise_rms

# Generation
for i in range(max_num):
    # randomly match audio from two person
    s1_all_sound_files = librosa.util.find_files(s1_path, ext=data_suffix)
    s2_all_sound_files = librosa.util.find_files(s2_path, ext=data_suffix)
    if len(s1_all_sound_files) == 0 or len(s2_all_sound_files) == 0:
        continue
    else:
        data_dict = {}
        s1_choosed_file = np.random.choice(s1_all_sound_files)
        s2_choosed_file = np.random.choice(s2_all_sound_files)

        # Read in two files and compare the power of the two audios
        x1, fs_x1 = sf.read(s1_choosed_file)
        x2, fs_x2 = sf.read(s2_choosed_file)

        x1_t, index1 = librosa.effects.trim(x1, top_db=30)
        x2_t, index2 = librosa.effects.trim(x2, top_db=30)

        start1 = np.random.randint(max(1, len(x1_t)-seq_len_temporal))
        start2 = np.random.randint(max(1, len(x2_t)-seq_len_temporal))

        x1_seg = x1[start1:(start1+seq_len_temporal)]
        x2_seg = x2[start2:(start2+seq_len_temporal)]
        
        # To examinate if there is still silence in the flute and speech audio
        x1_seg_trim, _ = librosa.effects.trim(x1_seg, top_db=30)
        if len(x1_seg_trim) != len(x1_seg):
            continue

        x2_seg_trim, _ = librosa.effects.trim(x2_seg, top_db=30)
        if len(x2_seg_trim) != len(x2_seg):
            continue

        if len(x1_seg) != len(x2_seg):
            continue

        clean_rms = cal_rms(x2_seg)
        noise_rms = cal_rms(x1_seg)
        adjusted_noise_rms = cal_adjusted_rms(clean_rms, snr)
        adjusted_noise_amp = x1_seg * (adjusted_noise_rms / noise_rms)
        mixed_amp = x2_seg + adjusted_noise_amp

        data_info = []
        data_info += 'clean_rms: %.4f \n' % clean_rms
        data_info += 'noise_rms: %.4f \n' % noise_rms
        data_info += 'adjust_ratio: %.4f \n' % adjusted_noise_rms
        data_info += 'adjust_noise_rms: %.4f \n' % cal_rms(adjusted_noise_amp)

        try:
            x_ref_s1 = adjusted_noise_amp
            x_est = np.ones_like(x_ref_s1)
            x_ref_s2 = x2_seg
            pesq(16000, x_ref_s1, x_est, mode='wb')
            pesq(16000, x_ref_s2, x_est, mode='wb')
            data_dict['x_t_mixed'] = mixed_amp
            data_dict['x1_t_gt'] = adjusted_noise_amp
            data_dict['x2_t_gt'] = x2_seg
            save_name = '{}_{}_{}_{}_{}dB'.format(os.path.split(s1_choosed_file)[-1].split('.')[0], start1, os.path.split(s2_choosed_file)[-1].split('.')[0], start2, snr)
            save_dir_npz = os.path.join(save_dir, 'npz_files')
            save_dir_example = os.path.join(save_dir, save_name)
            os.makedirs(save_dir_npz, exist_ok=True)
            os.makedirs(save_dir_example, exist_ok=True)

            save_dir_npz = os.path.join(save_dir_npz, '{}.npz'.format(save_name))
            np.savez_compressed(save_dir_npz, data=data_dict)
            save_path_mixture = os.path.join(save_dir_example, 'mixture.wav')
            sf.write(save_path_mixture, mixed_amp, fs_x2, format="wav")
            save_path_speech = os.path.join(save_dir_example, 'speech.wav')
            sf.write(save_path_speech, x2_seg, fs_x2, format="wav")
            save_path_flute = os.path.join(save_dir_example, 'flute.wav')
            sf.write(save_path_flute, adjusted_noise_amp, fs_x1, format="wav")

            data_info_file = os.path.join(save_dir_example, 'data_info.txt')
            with open(data_info_file, "w") as f:
                for info in data_info:
                    f.write(info)

            true_num += 1 
        except RuntimeError:
            continue

print('Generation finished, total data number: {}'.format(true_num))
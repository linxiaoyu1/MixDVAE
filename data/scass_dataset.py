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
import random
import numpy as np
import soundfile as sf
import librosa
import torch
import pickle
from torch.utils import data

def build_dataloader(cfg, data_type):

    # Load dataset params for WSJ0 subset
    dataset_name = cfg.get('DataFrame', 'dataset_name')
    batch_size = cfg.getint('DataFrame', 'batch_size')
    shuffle = cfg.getboolean('DataFrame', 'shuffle')
    num_workers = cfg.getint('DataFrame', 'num_workers')
    sequence_len = cfg.getint('DataFrame', 'sequence_len')
    data_suffix = cfg.get('DataFrame', 'suffix')
    use_random_seq = cfg.getboolean('DataFrame', 'use_random_seq')

    # Load and compute STFT parameters
    wlen_sec = cfg.getfloat('STFT', 'wlen_sec')
    hop_percent = cfg.getfloat('STFT', 'hop_percent')
    fs = cfg.getint('STFT', 'fs')
    zp_percent = cfg.getint('STFT', 'zp_percent')
    wlen = wlen_sec * fs
    wlen = np.int(np.power(2, np.ceil(np.log2(wlen)))) # pwoer of 2
    hop = np.int(hop_percent * wlen)
    nfft = wlen + zp_percent * wlen
    win = np.sin(np.arange(0.5, wlen+0.5) / wlen * np.pi)
    trim = cfg.getboolean('STFT', 'trim')

    STFT_dict = {}
    STFT_dict['fs'] = fs
    STFT_dict['wlen'] = wlen
    STFT_dict['hop'] = hop
    STFT_dict['nfft'] = nfft
    STFT_dict['win'] = win
    STFT_dict['trim'] = trim

    # Generate dataset
    if data_type == 'vem':
        vem_data_dir = cfg.get('User', 'vem_data_dir')
        file_list = os.listdir(vem_data_dir)
        dataset = SpeechSequencesVEM(file_dir=vem_data_dir, file_list=file_list, sequence_len=sequence_len,
                                        STFT_dict=STFT_dict, shuffle=shuffle, name=dataset_name)
    elif data_type == 'dvae_eval':
        eval_data_dir = cfg.get('User', 'eval_data_dir')
        file_list = librosa.util.find_files(eval_data_dir, ext=data_suffix)
        dataset = SpeechSequencesEvaluation(file_list=file_list, sequence_len=sequence_len,
                                        STFT_dict=STFT_dict, shuffle=shuffle, name=dataset_name)
    else:
        if data_type == 'dvae_train':
            train_data_dir = cfg.get('User', 'train_data_dir')        
            file_list = librosa.util.find_files(train_data_dir, ext=data_suffix)
            if use_random_seq:
                dataset = SpeechSequencesRandom(file_list=file_list, sequence_len=sequence_len,
                                                STFT_dict=STFT_dict, shuffle=shuffle, name=dataset_name)
            else:
                dataset = SpeechSequencesFull(file_list=file_list, sequence_len=sequence_len,
                                                STFT_dict=STFT_dict, shuffle=shuffle, name=dataset_name)
        elif data_type == 'dvae_val':
            val_data_dir = cfg.get('User', 'val_data_dir')
            file_list = librosa.util.find_files(val_data_dir, ext=data_suffix)
            dataset = SpeechSequencesFull(file_list=file_list, sequence_len=sequence_len,
                                                STFT_dict=STFT_dict, shuffle=shuffle, name=dataset_name)
        else:
            raise ValueError('Invalid data type!')
        

    sample_num = dataset.__len__()

    # Create dataloader
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, 
                                                shuffle=shuffle, num_workers=num_workers)

    return dataloader, sample_num


class SpeechSequencesFull(data.Dataset):
    """
    Customize a dataset of speech sequences for Pytorch
    at least the three following functions should be defined.
    """
    def __init__(self, file_list, sequence_len, STFT_dict, shuffle, name='WSJ0'):

        super().__init__()

        # STFT parameters
        self.fs = STFT_dict['fs']
        self.nfft = STFT_dict['nfft']
        self.hop = STFT_dict['hop']
        self.wlen = STFT_dict['wlen']
        self.win = STFT_dict['win']
        self.trim = STFT_dict['trim']
        
        # data parameters
        self.file_list = file_list
        self.sequence_len = sequence_len
        self.name = name
        self.shuffle = shuffle

        self.compute_len()


    def compute_len(self):

        self.valid_seq_list = []

        for wavfile in self.file_list:

            x, fs_x = sf.read(wavfile)
            if self.fs != fs_x:
                raise ValueError('Unexpected sampling rate')
            
            # remove beginning and ending silence
            if self.trim and ('TIMIT' in self.name):
                path, file_name = os.path.split(wavfile)
                path, speaker = os.path.split(path)
                path, dialect = os.path.split(path)
                path, set_type = os.path.split(path)
                with open(os.path.join(path, set_type, dialect, speaker, file_name[:-4] + '.PHN'), 'r') as f:
                    first_line = f.readline() # Read the first line
                    for last_line in f: # Loop through the whole file reading it all
                        pass
                if not('#' in first_line) or not('#' in last_line):
                    raise NameError('The first of last lines of the .phn file should contain #')
                ind_beg = int(first_line.split(' ')[1])
                ind_end = int(last_line.split(' ')[0])
            elif self.trim:
                _, (ind_beg, ind_end) = librosa.effects.trim(x, top_db=30)
            else:
                ind_beg = 0
                ind_end = len(x)


            # Check valid wav files
            seq_length = (self.sequence_len - 1) * self.hop
            file_length = ind_end - ind_beg 
            n_seq = (1 + int(file_length / self.hop)) // self.sequence_len
            for i in range(n_seq):
                seq_start = i * seq_length + ind_beg
                seq_end = (i + 1) * seq_length + ind_beg
                seq_info = (wavfile, seq_start, seq_end)
                self.valid_seq_list.append(seq_info)

        if self.shuffle:
            random.shuffle(self.valid_seq_list)


    def __len__(self):
        """
        arguments should not be modified
        Return the total number of samples
        """
        return len(self.valid_seq_list)


    def __getitem__(self, index):
        """
        input arguments should not be modified
        torch data loader will use this function to read ONE sample of data from a list that can be indexed by
        parameter 'index'
        """
        
        # Read wav files
        wavfile, seq_start, seq_end = self.valid_seq_list[index]
        x, fs_x = sf.read(wavfile)

        # Sequence tailor
        x = x[seq_start:seq_end]

        # # Normalize sequence
        x = x/np.max(np.abs(x))

        # STFT transformation
        audio_spec = librosa.stft(x, n_fft=self.nfft, hop_length=self.hop, win_length=self.wlen, window=self.win)

        # Square of magnitude
        sample = np.abs(audio_spec) ** 2
        # sample = sample/np.max(np.abs(sample))
        sample = torch.from_numpy(sample.astype(np.float32))

        return sample
    
class SpeechSequencesVEM(data.Dataset):
    """
    Customize a dataset of speech sequences for Pytorch
    at least the three following functions should be defined.
    """
    def __init__(self, file_dir, file_list, sequence_len, STFT_dict, shuffle, name='WSJ0'):

        super().__init__()
        
        # data parameters
        self.file_list = file_list
        self.file_dir = file_dir
        self.sequence_len = sequence_len
        self.name = name
        self.shuffle = shuffle
        
        # STFT parameters
        self.fs = STFT_dict['fs']
        self.nfft = STFT_dict['nfft']
        self.hop = STFT_dict['hop']
        self.wlen = STFT_dict['wlen']
        self.win = STFT_dict['win']
        self.trim = STFT_dict['trim']

        self.compute_len()


    def compute_len(self):

        self.valid_seq_list = []

        for file in self.file_list:
            file_path = os.path.join(self.file_dir, file)
            data_dict = np.load(file_path, allow_pickle=True)['data'].item()

            ind_beg = 0
            ind_end = len(data_dict['x_t_mixed'])
            
            # make borders
            length_temporal = (self.sequence_len - 1) * self.hop
            file_length = ind_end - ind_beg 
            n_seq = (1 + int(file_length / self.hop)) // self.sequence_len
            for i in range(n_seq):
                start_temporal = i * length_temporal + ind_beg
                end_temporal = (i + 1) * length_temporal + ind_beg
                seq_info = (file, start_temporal, end_temporal)
                self.valid_seq_list.append(seq_info)

        if self.shuffle:
            random.shuffle(self.valid_seq_list)


    def __len__(self):
        """
        arguments should not be modified
        Return the total number of samples
        """
        return len(self.valid_seq_list)


    def __getitem__(self, index):
        """
        input arguments should not be modified
        torch data loader will use this function to read ONE sample of data from a list that can be indexed by
        parameter 'index'
        """
        
        # Read files
        file, start_temporal, end_temporal = self.valid_seq_list[index]
        file_path = os.path.join(self.file_dir, file)
        data_dict = np.load(file_path, allow_pickle=True)['data'].item()

        # Sequence tailor
        x1_orig = data_dict['x1_t_gt'][start_temporal:end_temporal]
        x2_orig = data_dict['x2_t_gt'][start_temporal:end_temporal]
        x_mixed_orig = data_dict['x_t_mixed'][start_temporal:end_temporal]
 
        # Normalization
        normalization_factor = np.max(np.abs(x_mixed_orig))
        x1_norm = x1_orig / normalization_factor
        x2_norm = x2_orig / normalization_factor
        mixed_norm = x_mixed_orig / normalization_factor
        x_mixed = librosa.stft(mixed_norm, n_fft=self.nfft, hop_length=self.hop, win_length=self.wlen, window=self.win)
        audio_spec1_norm = librosa.stft(x1_norm, n_fft=self.nfft, hop_length=self.hop, win_length=self.wlen, window=self.win)
        power1_norm = np.abs(audio_spec1_norm) ** 2
        audio_spec2_norm = librosa.stft(x2_norm, n_fft=self.nfft, hop_length=self.hop, win_length=self.wlen, window=self.win)
        power2_norm = np.abs(audio_spec2_norm) ** 2
            
        sample_dict = {}
        sample_dict['data_mixed'] = torch.from_numpy(x_mixed)
        sample_dict['data_mixed_t'] = torch.from_numpy(x_mixed_orig)
        sample_dict['x1_t_gt'] = torch.from_numpy(x1_orig)
        sample_dict['x2_t_gt'] = torch.from_numpy(x2_orig)
        sample_dict['x1_stft_gt'] = torch.from_numpy(audio_spec1_norm)
        sample_dict['x2_stft_gt'] = torch.from_numpy(audio_spec2_norm)
        sample_dict['x1_power_gt'] = torch.from_numpy(power1_norm.astype(np.float32))
        sample_dict['x2_power_gt'] = torch.from_numpy(power2_norm.astype(np.float32))
        sample_dict['normalization_factor'] = normalization_factor

        return sample_dict

class SpeechSequencesEvaluation(data.Dataset):
    """
    Customize a dataset of speech sequences for Pytorch
    at least the three following functions should be defined.
    """
    def __init__(self, file_list, sequence_len, STFT_dict, shuffle, name='WSJ0'):

        super().__init__()

        # STFT parameters
        self.fs = STFT_dict['fs']
        self.nfft = STFT_dict['nfft']
        self.hop = STFT_dict['hop']
        self.wlen = STFT_dict['wlen']
        self.win = STFT_dict['win']
        self.trim = STFT_dict['trim']
        
        # data parameters
        self.file_list = file_list
        self.sequence_len = sequence_len
        self.name = name
        self.shuffle = shuffle

        self.compute_len()


    def compute_len(self):

        self.valid_seq_list = []

        for wavfile in self.file_list:

            x, fs_x = sf.read(wavfile)
            if self.fs != fs_x:
                raise ValueError('Unexpected sampling rate')
            
            # remove beginning and ending silence
            if self.trim and ('TIMIT' in self.name):
                path, file_name = os.path.split(wavfile)
                path, speaker = os.path.split(path)
                path, dialect = os.path.split(path)
                path, set_type = os.path.split(path)
                with open(os.path.join(path, set_type, dialect, speaker, file_name[:-4] + '.PHN'), 'r') as f:
                    first_line = f.readline() # Read the first line
                    for last_line in f: # Loop through the whole file reading it all
                        pass
                if not('#' in first_line) or not('#' in last_line):
                    raise NameError('The first of last lines of the .phn file should contain #')
                ind_beg = int(first_line.split(' ')[1])
                ind_end = int(last_line.split(' ')[0])
            elif self.trim:
                _, (ind_beg, ind_end) = librosa.effects.trim(x, top_db=30)
            else:
                ind_beg = 0
                ind_end = len(x)


            # Check valid wav files
            seq_length = (self.sequence_len - 1) * self.hop
            file_length = ind_end - ind_beg 
            n_seq = (1 + int(file_length / self.hop)) // self.sequence_len
            for i in range(n_seq):
                seq_start = i * seq_length + ind_beg
                seq_end = (i + 1) * seq_length + ind_beg
                seq_info = (wavfile, seq_start, seq_end)
                self.valid_seq_list.append(seq_info)

        if self.shuffle:
            random.shuffle(self.valid_seq_list)


    def __len__(self):
        """
        arguments should not be modified
        Return the total number of samples
        """
        return len(self.valid_seq_list)


    def __getitem__(self, index):
        """
        input arguments should not be modified
        torch data loader will use this function to read ONE sample of data from a list that can be indexed by
        parameter 'index'
        """
        
        # Read wav files
        wavfile, seq_start, seq_end = self.valid_seq_list[index]
        x, fs_x = sf.read(wavfile)

        # Sequence tailor
        x = x[seq_start:seq_end]

        # Normalize sequence
        x = x/np.max(np.abs(x))

        # STFT transformation
        audio_spec = librosa.stft(x, n_fft=self.nfft, hop_length=self.hop, win_length=self.wlen, window=self.win)

        # Square of magnitude
        sample = np.abs(audio_spec) ** 2
        sample = torch.from_numpy(sample.astype(np.float32))

        return sample, x
                
class SpeechSequencesRandom(data.Dataset):
    """
    Customize a dataset of speech sequences for Pytorch
    at least the three following functions should be defined.
    
    This is a quick speech sequence data loader which allow multiple workers
    """
    def __init__(self, file_list, sequence_len, STFT_dict, shuffle, name='WSJ0'):

        super().__init__()

        # STFT parameters
        self.fs = STFT_dict['fs']
        self.nfft = STFT_dict['nfft']
        self.hop = STFT_dict['hop']
        self.wlen = STFT_dict['wlen']
        self.win = STFT_dict['win']
        self.trim = STFT_dict['trim']
        
        # data parameters
        self.file_list = file_list
        self.sequence_len = sequence_len
        self.name = name
        self.shuffle = shuffle

        self.compute_len()


    def compute_len(self):

        self.valid_file_list = []

        for wavfile in self.file_list:

            x, fs_x = sf.read(wavfile)
            if self.fs != fs_x:
                raise ValueError('Unexpected sampling rate')

            # Silence clipping
            if self.trim:
                x, idx = librosa.effects.trim(x, top_db=30)

            # Check valid wav files
            seq_length = (self.sequence_len - 1) * self.hop
            if len(x) >= seq_length:
                self.valid_file_list.append(wavfile)

        if self.shuffle:
            random.shuffle(self.valid_file_list)


    def __len__(self):
        """
        arguments should not be modified
        Return the total number of samples
        """
        return len(self.valid_file_list)


    def __getitem__(self, index):
        """
        input arguments should not be modified
        torch data loader will use this function to read ONE sample of data from a list that can be indexed by
        parameter 'index'
        """
        
        # Read wav files
        wavfile = self.valid_file_list[index]
        x, fs_x = sf.read(wavfile)

        # Silence clipping
        if self.trim and ('TIMIT' in self.name):
            path, file_name = os.path.split(wavfile)
            path, speaker = os.path.split(path)
            path, dialect = os.path.split(path)
            path, set_type = os.path.split(path)
            with open(os.path.join(path, set_type, dialect, speaker, file_name[:-4] + '.PHN'), 'r') as f:
                first_line = f.readline() # Read the first line
                for last_line in f: # Loop through the whole file reading it all
                    pass
            if not('#' in first_line) or not('#' in last_line):
                raise NameError('The first of last lines of the .phn file should contain #')
            ind_beg = int(first_line.split(' ')[1])
            ind_end = int(last_line.split(' ')[0])
            x = x[ind_beg:ind_end]
        elif self.trim:
            x, _ = librosa.effects.trim(x, top_db=30)

        # Sequence tailor
        file_length = len(x)
        seq_length = (self.sequence_len - 1) * self.hop # sequence length in time domain
        start = np.random.randint(0, file_length - seq_length)
        end = start + seq_length
        x = x[start:end]

        # Normalize sequence
        x = x/np.max(np.abs(x))

        # STFT transformation
        audio_spec = librosa.stft(x, n_fft=self.nfft, hop_length=self.hop, win_length=self.wlen, window=self.win)

        # Square of magnitude
        sample = np.abs(audio_spec) ** 2
        sample = torch.from_numpy(sample.astype(np.float32))

        return sample


        


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
import shutil
import sys
from configparser import ConfigParser
import torch
from utils.save_model import SaveLog
from utils.utils import get_basic_info, load_vem_pretrained_dvae, evaluation_separation_temp, evaluation_mixture
from data.scass_dataset import build_dataloader
from models.vem_scass import VEM
import numpy as np
import pickle
import librosa
import soundfile as sf
import matplotlib.pyplot as plt
import librosa.display


def train(cfg_file):
    # Read the config file
    if not os.path.isfile(cfg_file):
        raise ValueError('Invalid config file path')
    cfg = ConfigParser()
    cfg.read(cfg_file)

    # Create save log directory
    save_log = SaveLog(cfg)
    save_dir = save_log.save_dir

    # Save config file
    save_cfg_path = os.path.join(save_dir, 'config.ini')
    shutil.copy(cfg_file, save_cfg_path)

    # Print basic information
    use_cuda = cfg.getboolean('Training', 'use_cuda')
    device = 'cuda' if torch.cuda.is_available() and use_cuda else 'cpu'

    basic_info = get_basic_info(device)
    save_log.print_info(basic_info)
    for info in basic_info:
        print('%s' % info)

    # Create and initialize model
    pretrained_dvae = load_vem_pretrained_dvae(cfg, device)
    vem_model = VEM(cfg, device, pretrained_dvae, save_log)

    # Set module.training = True if finetune during E-Z step.
    finetune = cfg.getboolean('Training', 'finetune')
    if finetune:
        for model in pretrained_dvae:
            model.train()
            torch.autograd.set_detect_anomaly(True)

    # Print model information
    for model in pretrained_dvae:
        model_info = model.get_info()
        save_log.print_info(model_info)
        for info in model_info:
            print('%s' % info)

    # Load data
    vem_data_loader, vem_data_size = build_dataloader(cfg, data_type='vem')

    # Print data information
    data_info = []
    data_info.append('========== DATA INFO ==========')
    data_info.append('Separation data: %s' % vem_data_size)
    save_log.print_info(data_info)
    for info in data_info:
        print('%s' % info)

    # Load and compute STFT parameters
    wlen_sec = cfg.getfloat('STFT', 'wlen_sec')
    hop_percent = cfg.getfloat('STFT', 'hop_percent')
    fs = cfg.getint('STFT', 'fs')
    zp_percent = cfg.getint('STFT', 'zp_percent')
    wlen = wlen_sec * fs
    wlen = int(np.power(2, np.ceil(np.log2(wlen)))) # pwoer of 2
    hop = int(hop_percent * wlen)
    nfft = wlen + zp_percent * wlen
    win = np.sin(np.arange(0.5, wlen+0.5) / wlen * np.pi)
    trim = cfg.getboolean('STFT', 'trim')

    STFT_dict = {}
    STFT_dict['fs'] = fs
    STFT_dict['wlen'] = wlen
    STFT_dict['hop'] = hop
    STFT_dict['nfft'] = nfft
    STFT_dict['win'] = win

    # Start separation
    print('Start separation...')
    total_iter = int(cfg.get('VEM', 'N_iter_total'))
    num_source = int(cfg.get('VEM', 'num_source'))
    save_eval_path = os.path.join(save_dir, 'separation_eval')
    os.makedirs(save_eval_path, exist_ok=True)
    num_metrics = 4
    eval_metrics_all_data = np.zeros((num_source, 1, total_iter, num_metrics))
    eval_metrics_all_mixture = np.zeros((num_source, 1, num_metrics))
    eval_metrics_all_IBM = np.zeros((num_source, 1, num_metrics))
    iter_num = 0
    unormal_index_list = np.zeros(1)
    for batch_idx, data in enumerate(vem_data_loader):
        batch_size = data['data_mixed'].shape[0]

        print('batch {}\n'.format(batch_idx))
        mu_phi_s_iter = vem_model.model_training(data)
        eval_metrics_mixture = evaluation_mixture(data, STFT_dict)

        x1_power_gt = data['x1_power_gt'].to(device)
        x2_power_gt = data['x2_power_gt'].to(device)
        s_gt_power = torch.cat((x1_power_gt.unsqueeze(0), x2_power_gt.unsqueeze(0)), 0)

        # Evaluate the separation performance with audio metrics
        eval_metrics_batch, eval_metrics_mixture, eval_metrics_IBM, unormal_index = evaluation_separation_temp([data['x1_t_gt'], data['x2_t_gt']], s_gt_power, mu_phi_s_iter, data['data_mixed_t'], data['data_mixed'], STFT_dict)

        eval_metrics_all_data = np.concatenate((eval_metrics_all_data, eval_metrics_batch), axis=1)
        eval_metrics_all_mixture = np.concatenate((eval_metrics_all_mixture, eval_metrics_mixture), axis=1)
        eval_metrics_all_IBM = np.concatenate((eval_metrics_all_IBM, eval_metrics_IBM), axis=1)
        unormal_index = unormal_index + iter_num
        unormal_index_list = np.concatenate((unormal_index_list, unormal_index))

        iter_num += batch_size

    eval_metrics_all_data = eval_metrics_all_data[:, 1:, :, :]
    eval_metrics_all_mixture = eval_metrics_all_mixture[:, 1:, :]
    eval_metrics_all_IBM = eval_metrics_all_IBM[:, 1:, :]
    unormal_index_list = unormal_index_list[1:]
    with open(os.path.join(save_dir, 'evaluation.pkl'), 'wb') as file:
        pickle.dump(eval_metrics_all_data, file)
    with open(os.path.join(save_dir, 'evaluation_mixture.pkl'), 'wb') as file:
        pickle.dump(eval_metrics_all_mixture, file)
    with open(os.path.join(save_dir, 'evaluation_IBM.pkl'), 'wb') as file:
        pickle.dump(eval_metrics_all_IBM, file)
    with open(os.path.join(save_dir, 'unormal_index.pkl'), 'wb') as file:
        pickle.dump(unormal_index_list, file)  

if __name__ == '__main__':
    if len(sys.argv) == 2:
        cfg_file = sys.argv[1]
        train(cfg_file)
    else:
        print('Error: Please indicate config file path')
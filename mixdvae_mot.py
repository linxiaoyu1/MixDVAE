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
import numpy as np
from configparser import ConfigParser

import motmetrics as mm
import torch

from utils.save_model import SaveLog
from utils.utils import get_basic_info, load_vem_pretrained_dvae
from utils.utils import tracking_evaluation_onebatch
from data.mot_dataset import build_dataloader
from models.vem_mot import VEM


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
        pretrained_dvae.train()
        torch.autograd.set_detect_anomaly(True)

    # Print model information
    pretrained_dvae_info = pretrained_dvae.get_info()
    save_log.print_info(pretrained_dvae_info)
    for info in pretrained_dvae_info:
        print('%s' % info)

    # Load data
    vem_data_loader, vem_data_size = build_dataloader(cfg, data_type='vem')

    # Print data information
    data_info = []
    data_info.append('========== DATA INFO ==========')
    data_info.append('Tracking data: %s' % vem_data_size)
    save_log.print_info(data_info)
    for info in data_info:
        print('%s' % info)

    # Start tracking
    print('Start tracking...')
    total_iter = int(cfg.get('VEM', 'N_iter_total'))
    contain_gt = cfg.getboolean('DataFrame', 'contain_gt')
    normalize_range = np.array([int(i) for i in cfg.get('DataFrame', 'normalize_range').split(',')], dtype='float64').reshape(-1,4)
    acc_list = [[] for i in range(total_iter)]
    for idx, data in enumerate(vem_data_loader):
        print('batch {}\n'.format(idx))
        if contain_gt:
            data_obs = data['det'].to(device)
            data_gt = data['gt'].to('cpu')
            Eta_iter, x_mean_dvaeumot_iter, x_var_dvaeumot_iter = vem_model.model_training(data_obs)
            acc_list = tracking_evaluation_onebatch(data_gt, normalize_range, acc_list, Eta_iter, x_mean_dvaeumot_iter)
        
        else:
            data_obs = data.to(device)
            Eta_iter, x_mean_dvaeumot_iter, x_var_dvaeumot_iter = vem_model.model_training(data_obs)

    # Evaluate the tracking performance for all batches
    if contain_gt:
        summary_list = []
        mota_list = [[] for i in range(total_iter)]
        for iter_number in range(total_iter):
            mh = mm.metrics.create()
            name = ['sample_{}'.format(i) for i in range(vem_data_size)]
            summary = mh.compute_many(acc_list[iter_number], metrics=mm.metrics.motchallenge_metrics, names=name, generate_overall=True)
            mota_list[iter_number].append(summary.loc['OVERALL']['mota'])
            strsummary = mm.io.render_summary(
                summary,
                formatters=mh.formatters,
                namemap=mm.io.motchallenge_metric_names
            )
            summary_list.append(strsummary)
        save_log.save_evaluation(summary_list, mota_list, total_iter)

if __name__ == '__main__':
    if len(sys.argv) == 2:
        cfg_file = sys.argv[1]
        train(cfg_file)
    else:
        print('Error: Please indicate config file path')

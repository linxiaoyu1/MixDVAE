## DVAE-UMOT
## Copyright Inria
## Year 2022
## Contact : xiaoyu.lin@inria.fr

## DVAE-UMOT is free software: you can redistribute it and/or modify
## it under the terms of the GNU General Public License as published by
## the Free Software Foundation, either version 3 of the License, or
## (at your option) any later version.

## DVAE-UMOT is distributed in the hope that it will be useful,
## but WITHOUT ANY WARRANTY; without even the implied warranty of
## MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
## GNU General Public License for more details.
##
## You should have received a copy of the GNU General Public License
## along with this program, DVAE-UMOT.  If not, see <http://www.gnu.org/licenses/> and the LICENSE file.

# DVAE-UMOT has code derived from 
# (1) ArTIST, https://github.com/fatemeh-slh/ArTIST.
# (2) DVAE, https://github.com/XiaoyuBIE1994/DVAE, distributed under MIT License 2020 INRIA.

import datetime
import math
import os
from random import random
import socket
import motmetrics as mm
import numpy as np
import torch
import librosa
from utils.eval_metrics_scass import EvalMetrics
import models.srnn_dvae_vem_mot
import models.srnn_dvae_single_mot
import models.srnn_dvae_single_scass
import models.srnn_dvae_vem_scass

def get_basic_info(device):
    host_name = socket.gethostname()
    time = datetime.datetime.now().strftime("%Y-%m-%d-%Hh%M")

    basic_info = []
    basic_info.append('========== BASIC INFO ==========')
    basic_info.append('Hostname: %s ' % host_name)
    basic_info.append('Time: %s' % time)
    basic_info.append('Device for training: %s' % device)
    if device == 'cuda':
        basic_info.append('Cuda version: %s\n' % torch.version.cuda)

    return basic_info


def get_loss_info(epoch, epoch_iter, time, loss_dict):
    loss = 'epoch: {} iters: {} time: {:.2f}m '.format(epoch, epoch_iter, time)
    for k, v in loss_dict.items():
        loss += '%s: %.3f ' % (k, v)

    loss_info = [loss]

    return loss_info


def initialize_optimizer(cfg, model):
    lr = cfg.getfloat('Training', 'lr')
    return torch.optim.Adam(model.parameters(), lr=lr)

def create_ar_model(cfg, device, save_dir):
    model_name = cfg.get('Network', 'name')
    if 'single' in model_name:
        model = models.ar_model_single.ARMODEL(cfg=cfg, device=device)
    elif 'umot' in model_name:
        model = models.ar_model_umot.ARMODEL(cfg=cfg, device=device)
    else:
        print('No such model!')

    # Load saved model for continue training
    continue_train = cfg.getboolean('Training', 'continue_train')
    if continue_train:
        which_epoch = cfg.get('Training', 'which_epoch')
        if which_epoch is not None:
            save_filename = 'models/model_epoch_%s.pt' % which_epoch
            save_path = os.path.join(save_dir, save_filename)
            if not os.path.isfile(save_path):
                raise ValueError('%s not exits!' % save_path)
            else:
                model.load_state_dict(torch.load(save_path, map_location=device))
        else:
            print('No epoch specified, model will be trained from the recorded latest epoch.')
            save_filename = 'models/model_epoch_latest.pt'
            save_path = os.path.join(save_dir, save_filename)
            if not os.path.isfile(save_path):
                raise ValueError('%s not exits!' % save_path)
            else:
                model.load_state_dict(torch.load(save_path, map_location=device))

    return model.to(device)

def create_dvae_model(cfg, device, save_dir):
    task_name = cfg.get('User', 'task_name')
    if task_name == 'MOT':
        model = models.srnn_dvae_single_mot.SRNN(cfg=cfg, device=device)
    elif task_name == 'SC-ASS':
        model = models.srnn_dvae_single_scass.SRNN(cfg=cfg, device=device)      
    else:
        print('No such model!')

    # Load saved model for continue training
    continue_train = cfg.getboolean('Training', 'continue_train')
    if continue_train:
        which_epoch = cfg.get('Training', 'which_epoch')
        if which_epoch is not None:
            save_filename = 'models/model_epoch_%s.pt' % which_epoch
            save_path = os.path.join(save_dir, save_filename)
            if not os.path.isfile(save_path):
                raise ValueError('%s not exits!' % save_path)
            else:
                model.load_state_dict(torch.load(save_path, map_location=device))
        else:
            print('No epoch specified, model will be trained from the recorded latest epoch.')
            save_filename = 'models/model_epoch_latest.pt'
            save_path = os.path.join(save_dir, save_filename)
            if not os.path.isfile(save_path):
                raise ValueError('%s not exits!' % save_path)
            else:
                model.load_state_dict(torch.load(save_path, map_location=device))

    return model.to(device)

def load_vem_pretrained_dvae(cfg, device):
    task_name = cfg.get('User', 'task_name')
    if task_name == 'MOT':
        pretrained_dvae_path = cfg.get('Training', 'saved_dvae')
        pretrained_model = models.srnn_dvae_vem_mot.SRNN(cfg=cfg, device=device)
        pretrained_model.load_state_dict(torch.load(pretrained_dvae_path, map_location=device))
    elif task_name == 'SC-ASS':
        pretrained_dvae_s1_path = cfg.get('Training', 'saved_dvae_s1')
        pretrained_dvae_s2_path = cfg.get('Training', 'saved_dvae_s2')
        dvae_s1 = models.srnn_dvae_vem_scass.SRNN(cfg=cfg, device=device)
        dvae_s2 = models.srnn_dvae_vem_scass.SRNN(cfg=cfg, device=device)
        dvae_s1.load_state_dict(torch.load(pretrained_dvae_s1_path, map_location=device))   
        dvae_s2.load_state_dict(torch.load(pretrained_dvae_s2_path, map_location=device))
        pretrained_model = [dvae_s1, dvae_s2]

    return pretrained_model

def init_training_params(cfg, save_dir, train_data_loader):
    n_epochs = cfg.getint('Training', 'n_epochs')
    early_stop_patience = cfg.getint('Training', 'early_stop_patience')

    iter_file_path = os.path.join(save_dir, 'iter.txt')
    start_epoch, epoch_iter = 1, 0
    continue_train = cfg.getboolean('Training', 'continue_train')
    if continue_train:
        which_epoch = cfg.get('Training', 'which_epoch')
        if which_epoch is not None:
            print('Resuming from epoch %s' % which_epoch)
        else:
            if os.path.exists(iter_file_path):
                start_epoch, epoch_iter = np.loadtxt(iter_file_path , delimiter=',', dtype=int)
            print('Resuming from epoch %d at iteration %d' % (start_epoch, epoch_iter))

    total_steps = (start_epoch - 1) * len(train_data_loader) + epoch_iter

    return n_epochs, early_stop_patience, \
           total_steps, start_epoch, epoch_iter, iter_file_path

def tracking_evaluation_onebatch(gt_seq, normalize_range, acc_list, eta_iter, x_mean_vem_iter):
    total_iter = eta_iter.shape[0]
    num_source = eta_iter.shape[1]
    batch_size = eta_iter.shape[2]
    seq_len = eta_iter.shape[3]
    num_obs = eta_iter.shape[4]
    gt_seq = gt_seq * np.repeat(np.reshape((normalize_range[0] - normalize_range[1]), (1, -1)), batch_size * seq_len * num_source, axis=0).reshape(batch_size, seq_len, num_source, -1) + np.repeat(np.reshape(normalize_range[1], (1,-1)), batch_size * seq_len * num_source, axis=0).reshape(batch_size, seq_len, num_source, -1)

    for iter_number in range(total_iter):
        x_mean_vem = x_mean_vem_iter[iter_number].to('cpu')
        eta = eta_iter[iter_number].to('cpu')

        # GT labels
        gt_labels = np.expand_dims(np.arange(num_obs), 0)
        gt_labels = np.repeat(gt_labels, seq_len * batch_size, axis=0).reshape(batch_size, seq_len, num_obs)

        # Tracking labels
        tk_labels = np.argmax(np.transpose(np.array(eta[:, :, :, :]), (1, 2, 0, 3)), axis=2)

        # Transform gt coordinates to (x,y,w,h)
        w_gt = gt_seq[:, :, :, 2] - gt_seq[:, :, :, 0]
        h_gt = gt_seq[:, :, :, 1] - gt_seq[:, :, :, 3]
        gt_coordinates = np.array(gt_seq.permute(3, 0, 1, 2))
        gt_coordinates[2, :, :, :] = w_gt
        gt_coordinates[3, :, :, :] = h_gt
        gt_coordinates = np.transpose(gt_coordinates, (1, 2, 3, 0))

        # Transform tracking coordinates to (x,y,w,h)
        x_mean_vem = x_mean_vem.permute(1, 2, 0, 3)
        x_mean_vem = x_mean_vem * np.repeat(np.reshape((normalize_range[0] - normalize_range[1]), (1, -1)), batch_size * seq_len * num_source, axis=0).reshape(batch_size, seq_len, num_source, -1) + np.repeat(np.reshape(normalize_range[1], (1,-1)), batch_size * seq_len * num_source, axis=0).reshape(batch_size, seq_len, num_source, -1)
        w_tk = x_mean_vem[:, :, :, 2] - x_mean_vem[:, :, :, 0]
        h_tk = x_mean_vem[:, :, :, 1] - x_mean_vem[:, :, :, 3]
        tk_coordinates = np.array(x_mean_vem.permute(3, 0, 1, 2))
        tk_coordinates[2, :, :, :] = w_tk
        tk_coordinates[3, :, :, :] = h_tk
        tk_coordinates = np.transpose(tk_coordinates, (1, 2, 3, 0))

        distance_matrix_all = []
        for i in range(batch_size):
            for t in range(seq_len):
                distance_matrix = mm.distances.iou_matrix(gt_coordinates[i, t], tk_coordinates[i, t], max_iou=0.5)
                distance_matrix_all.append(distance_matrix)
        distance_matrix_all = np.array(distance_matrix_all)
        distance_matrix_all = distance_matrix_all.reshape(batch_size, seq_len, num_source, num_obs)

        for i in range(batch_size):
            acc = mm.MOTAccumulator(auto_id=True)
            for t in range(seq_len):
                acc.update(gt_labels[i, t], tk_labels[i, t], distance_matrix_all[i, t])
            acc_list[iter_number].append(acc)

    return acc_list

def tracking_evaluation_onebatch_KF(gt_seq, normalize_range, acc_list, eta_iter, x_mean_vem_iter):
    total_iter = eta_iter.shape[0]
    num_source = eta_iter.shape[1]
    batch_size = eta_iter.shape[2]
    seq_len = eta_iter.shape[3]
    num_obs = eta_iter.shape[4]
    gt_seq = gt_seq * np.repeat(np.reshape((normalize_range[0] - normalize_range[1]), (1, -1)), batch_size * seq_len * num_source, axis=0).reshape(batch_size, seq_len, num_source, -1) + np.repeat(np.reshape(normalize_range[1], (1,-1)), batch_size * seq_len * num_source, axis=0).reshape(batch_size, seq_len, num_source, -1)

    for iter_number in range(total_iter):
        x_mean_vem = x_mean_vem_iter[iter_number].to('cpu')
        x_mean_vem = x_mean_vem * np.repeat(np.reshape((normalize_range[0] - normalize_range[1]), (1, -1)), batch_size * seq_len * num_source, axis=0).reshape(batch_size, seq_len, num_source, -1) + np.repeat(np.reshape(normalize_range[1], (1,-1)), batch_size * seq_len * num_source, axis=0).reshape(batch_size, seq_len, num_source, -1)
        eta = eta_iter[iter_number].to('cpu')

        # GT labels
        gt_labels = np.expand_dims(np.arange(num_obs), 0)
        gt_labels = np.repeat(gt_labels, seq_len * batch_size, axis=0).reshape(batch_size, seq_len, num_obs)

        # Tracking labels
        tk_labels = np.argmax(np.transpose(np.array(eta[:, :, :, :]), (1, 2, 0, 3)), axis=2)

        # Transform gt coordinates to (x,y,w,h)
        gt_coordinates = np.array(gt_seq)

        # Transform tracking coordinates to (x,y,w,h)
        tk_coordinates = np.array(x_mean_vem[:, :, :, :4].permute(1, 2, 0, 3))

        distance_matrix_all = []
        for i in range(batch_size):
            for t in range(seq_len):
                distance_matrix = mm.distances.iou_matrix(gt_coordinates[i, t], tk_coordinates[i, t], max_iou=0.5)
                distance_matrix_all.append(distance_matrix)
        distance_matrix_all = np.array(distance_matrix_all)
        distance_matrix_all = distance_matrix_all.reshape(batch_size, seq_len, num_source, num_obs)

        for i in range(batch_size):
            acc = mm.MOTAccumulator(auto_id=True)
            for t in range(seq_len):
                acc.update(gt_labels[i, t], tk_labels[i, t], distance_matrix_all[i, t])
            acc_list[iter_number].append(acc)

    return acc_list

def evaluation_mixture(data, STFT_dict):
    eval_metrics = EvalMetrics(metric='all')
    mixture = data['data_mixed']
    gt_s1 = data['x1_t_gt']
    gt_s2 = data['x2_t_gt']
    data_gt = [gt_s1, gt_s2]
    num_source = 2
    batch_size = mixture.shape[0]
    num_metrics = 4
    eval_metrics_mixture = np.zeros((num_source, batch_size, num_metrics))
    for n in range(num_source):
        for idx in range(batch_size):
            try:
                mixture_temp = librosa.istft(mixture[idx].cpu().numpy(), hop_length=STFT_dict['hop'], win_length=STFT_dict['wlen'], window=STFT_dict['win'])
                score_rmse_mixture, score_sisdr_mixture, score_pesq_mixture, score_estoi_mixture = eval_metrics.eval(x_est=mixture_temp, x_ref=data_gt[n][idx].cpu().numpy(), fs_est=STFT_dict['fs'])
                eval_metrics_mixture[n, idx, :] = np.array([score_rmse_mixture, score_sisdr_mixture, score_pesq_mixture, score_estoi_mixture])
            except RuntimeError:
                continue    

    return eval_metrics_mixture 

def evaluation_separation_temp(data_gt, data_gt_power, data_separated, mixture_data_temp, mixture_stft, STFT_dict):
    eval_metrics = EvalMetrics(metric='all')

    if len(data_separated.shape) == 5:
        n_iter = data_separated.shape[0]
        num_source = data_separated.shape[1]
        batch_size = data_separated.shape[2]
        num_metrics = 4

        IBM_mask = 1 - torch.argmax(data_gt_power, axis=0)

        unormal_index_all_source = []
        eval_metrics_all = np.zeros((num_source, batch_size, n_iter, num_metrics))
        eval_metrics_mixture_all = np.zeros((num_source, batch_size, num_metrics))
        eval_metrics_IBM_all = np.zeros((num_source, batch_size, num_metrics))
        for n in range(num_source):
            unormal_index_source = []
            for idx in range(batch_size):
                for i in range(n_iter):
                    try:
                        separate_temp = librosa.istft(data_separated[i, n, idx].permute(1,0).cpu().numpy(), hop_length=STFT_dict['hop'], win_length=STFT_dict['wlen'], window=STFT_dict['win'])
                        score_rmse, score_sisdr, score_pesq, score_estoi = eval_metrics.eval(x_est=separate_temp, x_ref=data_gt[n][idx].cpu().numpy(), fs_est=STFT_dict['fs'])
                        eval_metrics_all[n, idx, i, :] = np.array([score_rmse, score_sisdr, score_pesq, score_estoi])
                    except RuntimeError:
                        unormal_index_source.append(idx)
                        eval_metrics_all[n, idx, i, :] = np.array([np.nan, np.nan, np.nan, np.nan])
                        continue
                
                try:
                    s_IBM_stft = torch.abs(n - IBM_mask[idx]).cpu() * mixture_stft[idx].cpu()
                    s_IBM_temp = librosa.istft(s_IBM_stft.cpu().numpy(), hop_length=STFT_dict['hop'], win_length=STFT_dict['wlen'], window=STFT_dict['win'])
                    score_rmse_ibm, score_sisdr_ibm, score_pesq_ibm, score_estoi_ibm = eval_metrics.eval(x_est=s_IBM_temp, x_ref=data_gt[n][idx].cpu().numpy(), fs_est=STFT_dict['fs'])
                    score_rmse_mixture, score_sisdr_mixture, score_pesq_mixture, score_estoi_mixture = eval_metrics.eval(x_est=mixture_data_temp[idx].cpu().numpy(), x_ref=data_gt[n][idx].cpu().numpy(), fs_est=STFT_dict['fs'])
                    eval_metrics_mixture_all[n, idx, :] = np.array([score_rmse_mixture, score_sisdr_mixture, score_pesq_mixture, score_estoi_mixture])
                    eval_metrics_IBM_all[n, idx, :] = np.array([score_rmse_ibm, score_sisdr_ibm, score_pesq_ibm, score_estoi_ibm])
                except RuntimeError:
                    unormal_index_source.append(idx)
                    eval_metrics_mixture_all[n, idx, :] = np.array([np.nan, np.nan, np.nan, np.nan])
                    eval_metrics_IBM_all[n, idx, :] = np.array([np.nan, np.nan, np.nan, np.nan])
                    continue
            unormal_index_all_source += (unormal_index_source)
        
        unormal_index = np.unique(unormal_index_all_source)
        
        # if len(unormal_index) != 0:
        #     eval_metrics_all = np.delete(eval_metrics_all, unormal_index, axis=1)
        #     eval_metrics_mixture_all = np.delete(eval_metrics_mixture_all, unormal_index, axis=1)

        return eval_metrics_all, eval_metrics_mixture_all, eval_metrics_IBM_all, unormal_index




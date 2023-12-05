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

import os
import pickle

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter


class SaveLog():
    def __init__(self, cfg):
        self.cfg = cfg
        self.save_dir = self.create_save_directory()
        self.log_dir = self.create_log_file()
        self.log_file = self.create_log_file()
        self.tf_path = os.path.join(self.save_dir, 'tensorboard')
        self.tf_path_train = os.path.join(self.tf_path, 'training')
        self.tf_path_val = os.path.join(self.tf_path, 'val')
        self.tf_path_mixdvae = os.path.join(self.tf_path, 'mixdvae')
        self.summary_writer_training = SummaryWriter(self.tf_path_train)
        self.summary_writer_val = SummaryWriter(self.tf_path_val)
        self.summary_writer_mixdvae = SummaryWriter(self.tf_path_mixdvae)

    def create_save_directory(self):
        save_root = self.cfg.get('User', 'save_root')
        model_name = self.cfg.get('Network', 'name')
        dataset_name = self.cfg.get('DataFrame', 'dataset_name')
        directory_name = '{}_{}'.format(dataset_name, model_name)
        save_dir = os.path.join(save_root, directory_name)
        if not(os.path.isdir(save_dir)):
            os.makedirs(save_dir)

        return save_dir

    def save_config_file(self):
        save_path = os.path.join(self.save_dir, 'config.ini')
        with open(save_path, 'w') as configfile:
            self.cfg.write(configfile)

    def create_log_file(self):
        log_file = os.path.join(self.save_dir, 'log.txt')
        with open(log_file, "w") as f:
            f.write('Experiment Log\n')

        return log_file

    def print_info(self, info_list):
        with open(self.log_file, "a") as f:
            for info in info_list:
                f.write('%s\n' % info)

    def plot_current_training_loss(self, loss_dict, step):
        for k, v in loss_dict.items():
            self.summary_writer_training.add_scalar(k, v, step)

    def plot_current_val_loss(self, loss_dict, step):
        for k, v in loss_dict.items():
            self.summary_writer_val.add_scalar(k, v, step)

    def plot_mixdvae_dvae_loss(self, loss_dict, step):
        for k, v in loss_dict.items():
            self.summary_writer_mixdvae.add_scalar(k, v, step)

    def save_mixdvae_results(self, file_name, tracking_results):
        self.results_save_path = os.path.join(self.save_dir, 'tracking_results')
        if not(os.path.isdir(self.results_save_path)):
            os.makedirs(self.results_save_path)
        with open(os.path.join(self.results_save_path, '{}.pkl'.format(file_name)), 'wb') as file:
            pickle.dump(tracking_results, file)

    def save_KF_results(self, batch_idx, results_list):
        self.results_save_path = os.path.join(self.save_dir, 'Results_VEM_initphidiag_batch{}'.format(batch_idx))
        results_file_name = ['x_mean_mixdvae_iter.pkl', 'data_gt.pkl', 'data_obs.pkl']
        if not (os.path.isdir(self.results_save_path)):
            os.makedirs(self.results_save_path)
        for i in range(len(results_list)):
            with open(os.path.join(self.results_save_path, results_file_name[i]), 'wb') as file:
                pickle.dump(results_list[i].to('cpu'), file)

    def save_mixdvae_init_params(self, params_list, batch_idx):
        self.init_params_path = os.path.join(self.save_dir, 'InitParams_VEM_initphidiag_{}'.format(batch_idx))
        init_params_file_name = ['x_mean_mixdvae_init.pkl', 'x_var_mixdvae_init.pkl', 'x_sampled_init.pkl', 'Phi_init.pkl', 'Phi_inv_init.pkl', 'o.pkl']
        if not(os.path.isdir(self.init_params_path)):
            os.makedirs(self.init_params_path)
        for i in range(len(params_list)):
            with open(os.path.join(self.init_params_path, init_params_file_name[i]), 'wb') as file:
                pickle.dump(params_list[i].to('cpu'), file)

    def save_model_dvae(self, batch_idx, dvae_model):
        dvae_models_save_path = os.path.join(self.save_dir, 'DVAE_MODEL')
        if not(os.path.isdir(dvae_models_save_path)):
            os.makedirs(dvae_models_save_path)
        model_save_path = os.path.join(dvae_models_save_path, 'model_batch{}.pt'.format(batch_idx))
        torch.save(dvae_model.state_dict(), model_save_path)

    def save_model(self, epoch, epoch_iter, total_steps, model_state_dict, iter_file_path, end_of_epoch=False, save_best=False):
        save_latest_freq = self.cfg.getint('Training', 'save_latest_freq')
        save_epoch_freq = self.cfg.getint('Training', 'save_epoch_freq')
        save_models_file = os.path.join(self.save_dir, 'models')
        if not (os.path.isdir(save_models_file)):
            os.makedirs(save_models_file)
        if not end_of_epoch:
            if total_steps % save_latest_freq == 0:
                print('Saving the latest model epoch %d, total_steps %d' % (epoch, total_steps))
                save_latest_file = os.path.join(save_models_file, 'model_epoch_latest.pt')
                torch.save(model_state_dict, save_latest_file)
                np.savetxt(iter_file_path, (epoch, epoch_iter), delimiter=',', fmt='%d')

        else:
            if save_best:
                print('Saving the model with best validation loss at epoch %d, total_steps %d' % (epoch, total_steps))
                save_epoch_file = os.path.join(save_models_file, 'model_best.pt')
                torch.save(model_state_dict, save_epoch_file)
            if epoch % save_epoch_freq == 0:
                print('Saving the model at the end of epoch %d, total_steps %d' % (epoch, total_steps))
                save_latest_file = os.path.join(save_models_file, 'model_epoch_latest.pt')
                torch.save(model_state_dict, save_latest_file)
                save_epoch_file = os.path.join(save_models_file, 'model_epoch_%s.pt' % epoch)
                torch.save(model_state_dict, save_epoch_file)
                np.savetxt(iter_file_path, (epoch+1, 0), delimiter=',', fmt='%d')

    def save_evaluation(self, summary_list, mota_list, total_iter):
        eval_path = os.path.join(self.save_dir, 'evaluation_metrics.txt')
        mota_path = os.path.join(self.save_dir, 'mota_list.txt')
        with open(eval_path, "w") as text_file:
            for iter_number in range(total_iter):
                    text_file.write('#'*20)
                    text_file.write('Iteration {}'.format(iter_number))
                    text_file.write('#'*20)
                    text_file.write('\n')
                    text_file.write(summary_list[iter_number])
                    text_file.write('\n')
        np.savetxt(mota_path, mota_list, delimiter=',')









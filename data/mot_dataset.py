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
import pickle
import numpy as np
import torch.utils.data as data


class PositionDataset(data.Dataset):
    def __init__(self, cfg, data_type):
        super(PositionDataset, self).__init__()
        self.data_type = data_type
        if self.data_type == 'dvae_train':
            self.data_path = cfg.get('User', 'train_data_dir')
        elif self.data_type == 'dvae_val':
            self.data_path = cfg.get('User', 'val_data_dir')
        elif self.data_type == 'vem':
            self.data_path = cfg.get('User', 'vem_data_dir')
            self.contain_gt = cfg.getboolean('DataFrame', 'contain_gt')
        elif self.data_type == 'dvae_eval':
            self.data_path = cfg.get('User', 'eval_data_dir')
        self.name = cfg.get('DataFrame', 'dataset_name')
        self.normalize_range = np.array([int(i) for i in cfg.get('DataFrame', 'normalize_range').split(',')], dtype='float64').reshape(-1,4)
        self.shuffle_list = cfg.getboolean('DataFrame', 'shuffle_file_list')
        self.file_list = self.create_file_list()

    def create_file_list(self):
        file_list = os.listdir(self.data_path)
        if self.shuffle_list:
            np.random.shuffle(file_list)

        return file_list

    def __getitem__(self, index):
        data_path = os.path.join(self.data_path, self.file_list[index])
        with open(data_path, 'rb') as file:
            one_data = pickle.load(file)
        if self.data_type == 'vem':
            if self.contain_gt:
                num_obs = one_data['det'].shape[1]
                for i in range(num_obs):
                    one_data['det'][:, i, :] = (one_data['det'][:, i, :] - self.normalize_range[1]) / (self.normalize_range[0] - self.normalize_range[1])
                    one_data['gt'][:, i, :] = (one_data['gt'][:, i, :] - self.normalize_range[1]) / (
                                self.normalize_range[0] - self.normalize_range[1])
            else:
                num_obs = one_data.shape[1]
                for i in range(num_obs):
                    one_data[:, i, :] = (one_data[:, i, :] - self.normalize_range[1]) / (self.normalize_range[0] - self.normalize_range[1])
        else:
            one_data = (one_data - self.normalize_range[1]) / (self.normalize_range[0] - self.normalize_range[1])

        return one_data

    def __len__(self):
        return len(self.file_list)

def build_dataloader(cfg, data_type):
    batch_size = cfg.getint('DataFrame', 'batch_size')
    shuffle = cfg.getboolean('DataFrame', 'shuffle_samples_in_batch')
    num_worker = cfg.getint('DataFrame', 'num_workers')
    dataset = PositionDataset(cfg, data_type)
    data_size = dataset.__len__()
    dataloader = data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_worker)

    return dataloader, data_size



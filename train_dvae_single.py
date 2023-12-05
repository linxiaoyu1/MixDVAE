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

import datetime
import os
import shutil
import sys
from configparser import ConfigParser
import numpy as np
import torch
from utils.save_model import SaveLog
from utils.utils import get_basic_info, initialize_optimizer, create_dvae_model, init_training_params
from data import mot_dataset, scass_dataset
import time


def train(cfg_file):
    # Read the config file
    if not os.path.isfile(cfg_file):
        raise ValueError('Invalid config file path')
    cfg = ConfigParser()
    cfg.read(cfg_file)

    # Set random seed
    random_seed = cfg.getint('Training', 'random_seed')
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)

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
    model = create_dvae_model(cfg, device, save_dir)

    # Set module.training = True
    model.train()
    torch.autograd.set_detect_anomaly(True)

    # Print model information
    model_info = model.get_info()
    save_log.print_info(model_info)
    for info in model_info:
        print('%s' % info)

    # Create optimizer
    optimizer = initialize_optimizer(cfg, model)

    # Load data
    task_name = cfg.get('User', 'task_name')
    if task_name == 'MOT':
        train_data_loader, train_data_size = mot_dataset.build_dataloader(cfg, data_type='dvae_train')
        val_data_loader, val_data_size = mot_dataset.build_dataloader(cfg, data_type='dvae_val')
    elif task_name == 'SC-ASS':
        train_data_loader, train_data_size = scass_dataset.build_dataloader(cfg, data_type='dvae_train')
        val_data_loader, val_data_size = scass_dataset.build_dataloader(cfg, data_type='dvae_val')

    # Print data information
    data_info = []
    data_info.append('========== DATA INFO ==========')
    data_info.append('Training data: %s' % train_data_size)
    data_info.append('Validation data: %s' % val_data_size)
    save_log.print_info(data_info)
    for info in data_info:
        print('%s' % info)

    # Initialize training parameters 
    n_epochs, early_stop_patience, \
           total_steps, start_epoch, epoch_iter, iter_file_path = init_training_params(cfg, save_dir, train_data_loader)
    
    # Start training
    print('Start training...')
    st = time.time()
    best_val_loss = np.inf
    cpt_patience = 0
    cur_best_epoch = n_epochs
    best_state_dict = model.state_dict()
    schedule_sampling = cfg.getboolean('Training', 'schedule_sampling')

    for epoch in range(start_epoch, n_epochs):
        epoch_start_time = datetime.datetime.now()
        training_total_loss = 0
        training_recon_loss = 0
        training_KLD_loss = 0
        epoch_iter_number = 0
        if schedule_sampling:
            ss_start = cfg.getint('Training', 'ss_start')
            ss_end = cfg.getint('Training', 'ss_end')
            if epoch < ss_start:
                kl_warm = (epoch - 1)/ ss_start
                use_pred = 0
            elif epoch >= ss_start and epoch < ss_end:
                kl_warm = 1
                ss_step = epoch - ss_start
                use_pred = ss_step / (ss_end - ss_start)
            else:
                kl_warm = 1
                use_pred = 1            
            if epoch == 1:
                print('=====> KL warm up start')
            elif epoch == ss_start:
                print('=====> KL warm up end, schedule sampling start')
            elif epoch == ss_end:
                print('=====> schedule sampling end')
        else:
            ss_start = cfg.getint('Training', 'ss_start')
            if epoch < ss_start:
                kl_warm = (epoch - 1)/ ss_start
                use_pred = 0
            else:
                kl_warm = 1
                use_pred = 0
            if epoch == 1:
                print('=====> KL warm up start')
            elif epoch == ss_start:
                print('=====> KL warm up end')
        
        for idx, data in enumerate(train_data_loader, start=epoch_iter):
            batch_size = data.shape[0]
            total_steps += batch_size
            epoch_iter += batch_size
            epoch_iter_number += 1

            data = data.to(device)
            recon_data = model(data, compute_loss=True, kl_warm=kl_warm, use_pred=use_pred)

            loss_dict = model.loss
            optimizer.zero_grad()
            loss_dict['loss_tot'].backward()
            optimizer.step()

            training_total_loss += loss_dict['loss_tot'] * batch_size
            training_recon_loss += loss_dict['loss_recon'] * batch_size
            training_KLD_loss += loss_dict['loss_KLD'] * batch_size

            # Save latest model
            save_log.save_model(epoch, epoch_iter, total_steps, model.state_dict(), iter_file_path, end_of_epoch=False, save_best=False)

        training_total_loss = training_total_loss / train_data_size
        training_recon_loss = training_recon_loss / train_data_size
        training_KLD_loss = training_KLD_loss / train_data_size
        
        epoch_end_time = datetime.datetime.now()
        iter_time = (epoch_end_time - epoch_start_time).seconds / 60
        training_info = 'End of epoch {} \t training time: {:.2f}m \t training loss {:.4f} \t'\
            .format(epoch, iter_time, training_total_loss)
        # Display training loss
        save_log.plot_current_training_loss(loss_dict, total_steps)

        #Validation
        val_total_loss = 0
        val_recon_loss = 0
        val_KLD_loss = 0
        with torch.no_grad():
            for idx, val_data in enumerate(val_data_loader):
                batch_size = val_data.shape[0]
                val_data = val_data.to(device)
                val_data = torch.autograd.Variable(val_data)
                recon_data = model(val_data, compute_loss=True, kl_warm=kl_warm, use_pred=use_pred)

                loss_dict_val = model.loss
                val_total_loss += loss_dict_val['loss_tot'] * batch_size
                val_recon_loss += loss_dict_val['loss_recon'] * batch_size
                val_KLD_loss += loss_dict_val['loss_KLD']  * batch_size
            val_total_loss = val_total_loss / val_data_size
            val_recon_loss = val_recon_loss / val_data_size
            val_KLD_loss = val_KLD_loss / val_data_size
            avg_val_loss_dict = {'loss_tot': val_total_loss, 'loss_recon': val_recon_loss, 'loss_KLD': val_KLD_loss}
            save_log.plot_current_val_loss(avg_val_loss_dict, total_steps)
        torch.cuda.empty_cache()

        # Early stop patience
        if val_total_loss < best_val_loss:
            best_val_loss = val_total_loss
            cpt_patience = 0
            best_state_dict = model.state_dict()
            cur_best_epoch = epoch
        else:
            cpt_patience += 1

        # End of epoch
        training_info += 'val loss {:.4f}'.format(val_total_loss)
        training_info = [training_info]
        save_log.print_info(training_info)
        for info in training_info:
            print('%s' % info)

        # Stop training if early-stop triggers
        if cpt_patience == early_stop_patience:
            save_log.print_info(['Early stop patience achieved'])
            print('Early stop patience achieved')
            break

        # Save model for this epoch
        save_log.save_model(epoch, epoch_iter, total_steps, model.state_dict(), iter_file_path, end_of_epoch=True, save_best=False)

    # Save the final weights of network with the best validation loss
    save_log.save_model(cur_best_epoch, epoch_iter, total_steps, best_state_dict, iter_file_path, end_of_epoch=True, save_best=True)
    et = time.time()
    elapsed_time = et - st
    print("total training time {} s".format(elapsed_time))
    save_log.print_info("total training time {} s".format(elapsed_time))

if __name__ == '__main__':
    if len(sys.argv) == 2:
        cfg_file = sys.argv[1]
        train(cfg_file)
    else:
        print('Error: Please indicate config file path')




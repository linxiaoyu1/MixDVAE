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
import torch

class VEM():
    def __init__(self, cfg, device, dvae_model_list, save_log):
        self.device = device

        self.N_iter = cfg.getint('VEM', 'N_iter_total')
        self.batch_size = cfg.getint('DataFrame', 'batch_size')
        self.seq_len = cfg.getint('DataFrame', 'sequence_len')
        self.num_source = cfg.getint('VEM', 'num_source')
        self.num_freq = cfg.getint('VEM', 'num_freqency')
        self.finetune = cfg.getboolean('Training', 'finetune')

        self.dvae_models = dvae_model_list
        for model in self.dvae_models:
            model.to(self.device)

        self.z_dim = cfg.getint('Network', 'z_dim')

        self.lr = lr = cfg.getfloat('Training', 'lr')
        self.ratio_v_theta_x = cfg.getfloat('VEM', 'ratio_v_theta_x')
        self.optimizer_list = [torch.optim.Adam(dvae_model_list[0].parameters(), lr=lr), torch.optim.Adam(dvae_model_list[1].parameters(), lr=lr)]
        
        self.save_log = save_log

    def model_training(self, data_dict):
        ### Tensor dimensions
        ## Input
        # data_dict:
        # - data_mixed (complex): (batch_size, num_freq, seq_len)
        # - x1_t_gt/x2_t_gt : (batch_size, temporal_len)
        # - x1_power_gt/x2_power_gt : (batch_size, num_freq, seq_len)
        # - v_theta_x : (batch_size, num_freq, seq_len)
        
        ## Initialized tensors
        # s_power_init: (num_source, batch_size, seq_len, num_freq)
        # Eta_init: (num_source, batch_size, seq_len, num_freq)
        # v_theta_x: (batch_size, seq_len, num_freq)

        ## Recording tensors
        # Eta_iter: (N_iter, num_source, batch_size, seq_len, num_freq)
        # z_mean_inf_iter: (N_iter, num_source, batch_size, seq_len, z_dim)
        # z_logvar_inf_iter: (N_iter, num_source, batch_size, seq_len, z_dim)
        # mu_phi_s_iter (complex): (N_iter, num_source, batch_size, seq_len, num_freq)
        # v_phi_s_iter: (N_iter, num_source, batch_size, seq_len, num_freq)
        # v_theta_s_iter: (N_iter, num_source, batch_size, seq_len, num_freq)

        ## Tensors inside iteration
        # Eta_n: (batch_size, seq_len, num_freq)
        # z_mean_inf_n: (batch_size, seq_len, z_dim)
        # z_logvar_inf_n: (batch_size, seq_len, z_dim)
        # mu_phi_s_n (complex): (batch_size, seq_len, num_freq)
        # v_phi_s_n: (batch_size, seq_len, num_freq)
        # v_theta_s_n: (batch_size, seq_len, num_freq)

        # Parameters Initialization
        data_mixed = data_dict['data_mixed'].to(self.device).permute(0, 2, 1)
        mixed_power = torch.abs(data_mixed) ** 2
        mixed_power = mixed_power.float()
        batch_size = data_mixed.shape[0]
        v_theta_x = mixed_power * self.ratio_v_theta_x

        with torch.no_grad():
            recon_data_s1 = self.dvae_models[0].vem_initialization(mixed_power)
            recon_data_s2 = self.dvae_models[1].vem_initialization(mixed_power)

        s_power_init = torch.cat((recon_data_s1.detach().unsqueeze(0), recon_data_s2.detach().unsqueeze(0)), 0)
        Eta_init = 0.5 * torch.ones((self.num_source, batch_size, self.seq_len, self.num_freq)).to(self.device)
        
        # Initialization of recording tensors
        mu_phi_s_iter = torch.zeros((self.N_iter, self.num_source, batch_size, self.seq_len, self.num_freq), dtype=torch.cfloat).to(self.device)
        
        dict_last_iter = {'eta':Eta_init,  
                          'v_phi_s': torch.zeros(self.num_source, batch_size, self.seq_len, self.num_freq).to(self.device),
                          's_sampled_power': torch.zeros(self.num_source, batch_size, self.seq_len, self.num_freq).to(self.device)
                          }

        # Start VEM iterations
        for i in range(self.N_iter):
            iter_start = datetime.datetime.now()
            
            # E-S and E-Z Step
            es_start = datetime.datetime.now()
            loss_esez = torch.zeros(1).to(self.device)
            loss_recon = torch.zeros(1).to(self.device)
            loss_kld = torch.zeros(1).to(self.device)
            loss_dvae = torch.zeros(1).to(self.device)
            if self.finetune:
                for n in range(self.num_source):
                    if i == 0:
                        s_power = s_power_init[n, :, :, :]
                        Eta_n = Eta_init[n, :, :, :]
                    else:
                        s_power = torch.clone(dict_last_iter['s_sampled_power'][n, :, :, :])
                        Eta_n = dict_last_iter['eta'][n, :, :, :]

                    v_theta_s_n, v_phi_s_n, mu_phi_s_n, s_sampled_n, s_sampled_power_n, z_n = self.dvae_models[n](s_power,
                                                                                            Eta_n, v_theta_x, data_mixed, compute_loss=True)
                    loss_dict = self.dvae_models[n].loss
                    mu_phi_s_iter[i, n, :, :, :] = mu_phi_s_n.detach()
                    dict_last_iter['v_phi_s'][n, :, :, :] = v_phi_s_n.detach()
                    dict_last_iter['s_sampled_power'][n, :, :, :] = s_sampled_power_n.detach()
                    
                    loss_dvae += loss_dict['loss_tot'].detach()
                    loss_recon += loss_dict['loss_recon'].detach()
                    loss_kld += loss_dict['loss_KLD'].detach()

                    self.optimizer_list[n].zero_grad()
                    loss_dict['loss_tot'].backward()
                    self.optimizer_list[n].step()
            else:
                with torch.no_grad():
                    for n in range(self.num_source):
                        if i == 0:
                            s_power = s_power_init[n, :, :, :]
                            Eta_n = Eta_init[n, :, :, :]
                        else:
                            s_power = dict_last_iter['s_sampled_power'][n, :, :, :]
                            Eta_n = dict_last_iter['eta'][n, :, :, :]

                        v_theta_s_n, v_phi_s_n, mu_phi_s_n, s_sampled_n, s_sampled_power_n, z_n = self.dvae_models[n](s_power,
                                                                                                Eta_n, v_theta_x, data_mixed, compute_loss=True)

                        loss_dict = self.dvae_models[n].loss
                        mu_phi_s_iter[i, n, :, :, :] = mu_phi_s_n.detach()
                        dict_last_iter['v_phi_s'][n, :, :, :] = v_phi_s_n.detach()
                        dict_last_iter['s_sampled_power'][n, :, :, :] = s_sampled_power_n.detach()

                        loss_dvae += loss_dict['loss_tot'].detach()
                        loss_recon += loss_dict['loss_recon'].detach()
                        loss_kld += loss_dict['loss_KLD'].detach()

            loss_dvae = loss_dvae / self.num_source
            loss_recon = loss_recon / self.num_source
            loss_kld = loss_kld / self.num_source
            loss_qs = - torch.sum(0.5 * torch.log(dict_last_iter['v_phi_s'].view(batch_size * self.seq_len * self.num_source, self.num_freq)))
            loss_qs = loss_qs / (batch_size * self.seq_len * self.num_source)

            es_end = datetime.datetime.now()
            es_time = (es_end - es_start).seconds / 60
            print('E-S time {:.2f}m'.format(es_time))                
            
            # E-W Step
            ew_start = datetime.datetime.now()
            Eta_n_sum = torch.zeros(batch_size, self.seq_len, self.num_freq).to(self.device)
           
            for n in range(self.num_source):
                mu_phi_s_n = mu_phi_s_iter[i, n, :, :, :]
                v_phi_s_n = dict_last_iter['v_phi_s'][n, :, :, :]

                Eta_n = self.compute_eta_n(data_mixed, v_theta_x, mu_phi_s_n, v_phi_s_n)
                Eta_n_tosum = torch.clone(Eta_n)
                Eta_n_tosum[torch.isnan(Eta_n_tosum)] = 0
                Eta_n_sum += Eta_n_tosum
                dict_last_iter['eta'][n, :, :, :] = Eta_n
            Eta_n_sum = Eta_n_sum.expand(self.num_source, batch_size, self.seq_len, self.num_freq)
            dict_last_iter['eta'] = dict_last_iter['eta'] / Eta_n_sum
            Eta_for_loss = torch.clone(dict_last_iter['eta'])
            Eta_for_loss[torch.isnan(Eta_for_loss)] = 0
            loss_qw = torch.sum(Eta_for_loss * (Eta_for_loss + 0.0000000001).log()) / (self.num_source * batch_size * self.seq_len * self.num_freq)
            ew_end = datetime.datetime.now()
            ew_time = (ew_end - ew_start).seconds / 60
            print('E-W time {:.2f}m'.format(ew_time))

            # Save the results
            loss_qw_qs = self.compute_loss_qwqs(dict_last_iter['eta'], data_mixed, v_theta_x, mu_phi_s_iter[i, :, :, :, :], dict_last_iter['v_phi_s'])
            loss_elbo = loss_qw_qs + loss_dvae + loss_qw + loss_qs
            print('loss_elbo: {}'.format(loss_elbo))
            print('loss_qw_qs: {}'.format(loss_qw_qs))
            print('loss_dvae: {}'.format(loss_dvae))
            print('loss_qw: {}'.format(loss_qw))
            print('loss_qs: {}'.format(loss_qs))



        return mu_phi_s_iter

    def compute_eta_n(self, data_mixed, v_theta_x, mu_phi_s_n, v_phi_s_n):
        ### Tensor dimensions
        ## Inputs:
        # data_mixed (complex): (batch_size, seq_len, num_freq)
        # v_theta_x: (batch_size, seq_len, num_freq)
        # mu_phi_s_n (complex): (batch_size, seq_len, num_freq)
        # v_phi_s_n: (batch_size, seq_len, num_freq)

        # Output:
        # Eta_n: (batch_size, seq_len, num_freq)

        diff_data_mu_power = torch.abs((data_mixed - mu_phi_s_n)) ** 2
        Eta_n = torch.exp(- v_theta_x.log() - diff_data_mu_power/v_theta_x - v_phi_s_n/v_theta_x) + 0.0000000001

        return Eta_n

    def compute_loss_qwqs(self, Eta, data_mixed, v_theta_x, mu_phi_s, v_phi_s):
        batch_size = Eta.shape[1]
        loss_qwqs = torch.zeros(batch_size, self.seq_len).to(self.device)
        for n in range(self.num_source):
            mu_phi_s_n = mu_phi_s[n, :, :, :]
            v_phi_s_n = v_phi_s[n, :, :, :]
            diff_data_mu_power = torch.abs((data_mixed - mu_phi_s_n)) ** 2
            loss_qwqs_n = - Eta * (v_theta_x.log() + diff_data_mu_power/v_theta_x + v_phi_s_n/v_theta_x)
            loss_qwqs_n[torch.isnan(loss_qwqs_n)] = 0
            loss_qwqs += loss_qwqs_n.sum()

        loss_qwqs = torch.sum(loss_qwqs) / (self.num_source * batch_size * self.seq_len * self.num_freq)

        return loss_qwqs












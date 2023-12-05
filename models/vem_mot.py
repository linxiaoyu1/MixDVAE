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
    def __init__(self, cfg, device, dvae_model, save_log):
        self.device = device

        self.N_iter_total = cfg.getint('VEM', 'N_iter_total')
        self.num_source = cfg.getint('VEM', 'num_source')
        self.num_obs = cfg.getint('VEM', 'num_obs')
        self.std_ratio = cfg.getfloat('VEM', 'std_ratio')
        self.init_iter_number = cfg.getint('VEM', 'init_iter_number')
        self.init_subseq_len = cfg.getint('VEM', 'init_subseq_len')
        self.finetune = cfg.getboolean('Training', 'finetune')

        self.dvae_model = dvae_model.to(self.device)

        self.o_dim = cfg.getint('VEM', 'o_dim')
        self.x_dim = cfg.getint('Network', 'x_dim')
        self.z_dim = cfg.getint('Network', 'z_dim')

        self.lr = lr = cfg.getfloat('Training', 'lr')
        self.optimizer = torch.optim.Adam(self.dvae_model.parameters(), lr=lr)
        self.save_log = save_log

    def model_training(self, data_obs):
        ### Tensor dimensions
        ## Initialized tensors
        # x_mean_dvaeumot_init: (num_source, batch_size, seq_len, x_dim)
        # x_var_dvaeumot_init: (num_source, batch_size, seq_len, x_dim, x_dim)
        # Phi_init: (batch_size, seq_len, num_obs, o_dim, o_dim)

        ## Recording tensors
        # Eta_iter: (N_iter_total, num_source, batch_size, seq_len, num_obs)
        # x_mean_dvaeumot_iter: (N_iter_total, num_source, batch_size, seq_len, x_dim)
        # x_var_dvaeumot_iter: (N_iter_total, num_source, batch_size, seq_len, x_dim, x_dim)
        # Phi_iter: (N_iter_total, batch_size, seq_len, num_obs, o_dim, o_dim)
        # Phi_inv_iter: (N_iter_total, batch_size, seq_len, num_obs, o_dim, o_dim)

        ## Tensors inside iteration
        # Eta_n: (batch_size, seq_len, num_obs)
        # x_mean_dvaeumot_n: (batch_size, seq_len, x_dim)
        # x_var_dvaeumot_n: (batch_size, seq_len, x_dim, x_dim)
        # Phi: (batch_size, seq_len, num_obs, o_dim, o_dim)
        # Phi_inv: (batch_size, seq_len, num_obs, o_dim, o_dim)

        # data_obs: (batch_size, seq_len, num_obs, o_dim)

        # Parameters Initialization
        data_obs = data_obs.float()
        batch_size = data_obs.shape[0]
        seq_len = data_obs.shape[1]
        
        x_mean_dvaeumot_init, x_var_dvaeumot_init, Phi_init, Phi_inv_init = self.parameters_init_split(data_obs, iter_number=self.init_iter_number, split_len=self.init_subseq_len, std_ratio=self.std_ratio)

        # Initialization of recording tensors
        Eta_iter = torch.zeros(self.N_iter_total, self.num_source, batch_size, seq_len, self.num_obs).to(self.device)
        x_mean_dvaeumot_iter = torch.zeros(self.N_iter_total, self.num_source, batch_size, seq_len, self.x_dim).to(self.device)
        x_var_dvaeumot_iter = torch.zeros(self.N_iter_total, self.num_source, batch_size, seq_len, self.x_dim, self.x_dim).to(self.device)

        # Start VEM iterations
        for i in range(self.N_iter_total):
            iter_start = datetime.datetime.now()
            ew_start = datetime.datetime.now()
            Eta_n_sum = torch.zeros(batch_size, seq_len, self.num_obs).to(self.device)

            # E-W Step
            for n in range(self.num_source):
                if i == 0:
                    x_mean_dvaeumot_n = x_mean_dvaeumot_init[n, :, :, :]
                    x_var_dvaeumot_n = x_var_dvaeumot_init[n, :, :, :, :]

                else:
                    x_mean_dvaeumot_n = x_mean_dvaeumot_iter[i-1, n, :, :, :]
                    x_var_dvaeumot_n = x_var_dvaeumot_iter[i-1, n, :, :, :, :]
                Phi = Phi_init
                Phi_inv = Phi_inv_init
                
                Eta_n = self.compute_eta_n(data_obs, Phi, Phi_inv, x_mean_dvaeumot_n, x_var_dvaeumot_n)
                Eta_n_tosum = torch.clone(Eta_n)
                Eta_n_tosum[torch.isnan(Eta_n_tosum)] = 0
                Eta_n_sum += Eta_n_tosum
                Eta_iter[i, n, :, :, :] = Eta_n
            Eta_n_sum = Eta_n_sum.expand(self.num_source, batch_size, seq_len, self.num_obs)
            Eta_iter[i, :, :, :, :] = Eta_iter[i, :, :, :, :] / Eta_n_sum
            Eta_for_loss = torch.clone(Eta_iter[i, :, :, :, :])
            Eta_for_loss[torch.isnan(Eta_for_loss)] = 0
            loss_qw = torch.sum(Eta_for_loss * (Eta_for_loss + 0.0000000001).log()) / (self.num_source * batch_size * seq_len * self.num_obs)
            ew_end = datetime.datetime.now()
            ew_time = (ew_end - ew_start).seconds / 60
            print('E-W time {:.2f}m'.format(ew_time))

            # E-S and E-Z Step
            es_start = datetime.datetime.now()
            loss_esez = torch.zeros(1).to(self.device)
            loss_recon = torch.zeros(1).to(self.device)
            loss_kld = torch.zeros(1).to(self.device)
            loss_dvae = torch.zeros(1).to(self.device)
            if self.finetune:
                for n in range(self.num_source):
                    if i == 0:
                        x_dvaeumot_im1_n = x_mean_dvaeumot_init[n, :, :, :]
                    else:
                        x_dvaeumot_im1_n = x_mean_dvaeumot_iter[i-1, n, :, :, :]
                    Phi_inv = Phi_inv_init
                    Eta_n = Eta_iter[i, n, :, :, :]
                    frame0 = data_obs[:, 0, n, :]
                    x_mean_dvaeumot_n, x_var_dvaeumot_n = self.dvae_model(x_dvaeumot_im1_n, Eta_n, Phi_inv, data_obs, frame0, compute_loss=True)
                    loss_dict = self.dvae_model.loss
                    
                    loss_dvae += loss_dict['loss_tot'].detach()
                    loss_recon += loss_dict['loss_recon'].detach()
                    loss_kld += loss_dict['loss_KLD'].detach()
                    loss_esez += loss_dict['loss_tot']

                    x_mean_dvaeumot_iter[i, n, :, :, :] = x_mean_dvaeumot_n.detach()
                    x_var_dvaeumot_iter[i, n, :, :, :, :] = x_var_dvaeumot_n.detach()               
                self.optimizer.zero_grad()
                loss_esez.backward()
                self.optimizer.step()                
            else:
                with torch.no_grad():
                    for n in range(self.num_source):
                        if i == 0:
                            x_dvaeumot_im1_n = x_mean_dvaeumot_init[n, :, :, :]
                        else:
                            x_dvaeumot_im1_n = x_mean_dvaeumot_iter[i-1, n, :, :, :]
                        Phi_inv = Phi_inv_init
                        Eta_n = Eta_iter[i, n, :, :, :]
                        frame0 = data_obs[:, 0, n, :]
                        x_mean_dvaeumot_n, x_var_dvaeumot_n = self.dvae_model(x_dvaeumot_im1_n, Eta_n, Phi_inv, data_obs, frame0, compute_loss=True)
                        loss_dict = self.dvae_model.loss
                        
                        loss_dvae += loss_dict['loss_tot'].detach()
                        loss_recon += loss_dict['loss_recon'].detach()
                        loss_kld += loss_dict['loss_KLD'].detach()
                        loss_esez += loss_dict['loss_tot']

                        x_mean_dvaeumot_iter[i, n, :, :, :] = x_mean_dvaeumot_n.detach()
                        x_var_dvaeumot_iter[i, n, :, :, :, :] = x_var_dvaeumot_n.detach()

            loss_dvae = loss_dvae / self.num_source
            loss_recon = loss_recon / self.num_source
            loss_kld = loss_kld / self.num_source
            loss_qs = - torch.sum(0.5 * torch.log(torch.det(x_var_dvaeumot_iter[i, :, :, :, :, :].view(batch_size * seq_len * self.num_source, self.x_dim, self.x_dim))))
            loss_qs = loss_qs / (batch_size * seq_len * self.num_source)

            es_end = datetime.datetime.now()
            es_time = (es_end - es_start).seconds / 60
            print('E-S time {:.2f}m'.format(es_time))

            iter_end = datetime.datetime.now()
            iter_time = (iter_end - iter_start).seconds / 60
            print('Iter time {:.2f}m'.format(iter_time))

            # Print the losses
            loss_qw_qs = self.compute_loss_qwqs(Eta_iter[i, :, :, :, :], data_obs, Phi, Phi_inv, x_mean_dvaeumot_iter[i, :, :, :, :], x_var_dvaeumot_iter[i, :, :, :, :, :])
            loss_elbo = loss_qw_qs + loss_dvae + loss_qw + loss_qs
            print('loss_elbo: {}'.format(loss_elbo))
            print('loss_qw_qs: {}'.format(loss_qw_qs))
            print('loss_dvae: {}'.format(loss_dvae))
            print('loss_qw: {}'.format(loss_qw))
            print('loss_qs: {}'.format(loss_qs))

        return Eta_iter, x_mean_dvaeumot_iter, x_var_dvaeumot_iter

    def parameters_init_split(self, data_obs, iter_number=10, split_len=50, std_ratio=0.04):
        batch_size = data_obs.shape[0]
        seq_len = data_obs.shape[1]

        # Initialize the observation variance matrix Phi with the size of detection bounding boxes at the first frame
        variance_matrix = torch.zeros(batch_size, seq_len, self.num_obs, self.o_dim, self.o_dim).to(self.device)
        variance_matrix_inv = torch.zeros(batch_size, seq_len, self.num_obs, self.o_dim, self.o_dim).to(self.device)
        for i in range(batch_size):
            for j in range(self.num_obs):
                std_onesource = torch.zeros(self.o_dim)
                w = data_obs[i, 0, j, 2] - data_obs[i, 0, j, 0]
                h = data_obs[i, 0, j, 3] - data_obs[i, 0, j, 1]
                std_w = w * std_ratio
                std_h = h * std_ratio
                std_onesource[0] = std_w
                std_onesource[2] = std_w
                std_onesource[1] = std_h
                std_onesource[3] = std_h
                variance_matrix_onesource = torch.diag(torch.pow(std_onesource, 2))
                variance_matrix_onesource_inv = torch.inverse(variance_matrix_onesource)
                variance_matrix_onesource_seq = variance_matrix_onesource.expand(seq_len, self.o_dim, self.o_dim)
                variance_matrix_onesource_inv_seq = variance_matrix_onesource_inv.expand(seq_len, self.o_dim, self.o_dim)

                variance_matrix[i, :, j, :, :] = variance_matrix_onesource_seq
                variance_matrix_inv[i, :, j, :, :] = variance_matrix_onesource_inv_seq

        x_mean_dvaeumot_init = torch.zeros(self.num_source, batch_size, seq_len, self.x_dim).to(self.device)
        x_var_dvaeumot_init = torch.zeros(self.num_source, batch_size, seq_len, self.x_dim, self.x_dim).to(self.device)

        Phi_init = variance_matrix
        Phi_inv_init = variance_matrix_inv

        # Initialization by sub-sequences
        start_frame = 0
        while start_frame < seq_len:

            # Pre-Initialization
            x_mean_dvaeumot_init_split = torch.zeros(self.num_source, batch_size, split_len, self.x_dim).to(self.device)
            x_var_dvaeumot_init_split = torch.zeros(self.num_source, batch_size, split_len, self.x_dim, self.x_dim).to(
                self.device)
            Phi_init_split = variance_matrix[:, start_frame:start_frame+split_len, :, :, :]
            Phi_inv_init_split = variance_matrix_inv[:, start_frame:start_frame+split_len, :, :, :]

            # Initialize the sequence of s with frame 0
            for n in range(self.num_source):
                if start_frame == 0:
                    frame0 = data_obs[:, 0, n, :]
                else:
                    frame0 = x_mean_dvaeumot_iter_split[-1, n, :, -1, :]

                x_var_dvaeumot_init_n = Phi_init_split[:, :, n, :, :].permute(1, 0, 2, 3)
                x_mean_dvaeumot_init_n = frame0.expand(split_len, batch_size, self.x_dim)

                x_mean_dvaeumot_init_split[n, :, :, :] = x_mean_dvaeumot_init_n.permute(1, 0, 2).squeeze()
                x_var_dvaeumot_init_split[n, :, :, :, :] = x_var_dvaeumot_init_n.permute(1, 0, 2, 3).squeeze()

            x_mean_dvaeumot_init[:, :, start_frame:start_frame+split_len, :] = x_mean_dvaeumot_init_split
            x_var_dvaeumot_init[:, :, start_frame:start_frame+split_len, :, :] = x_var_dvaeumot_init_split

            data_obs_split = data_obs[:, start_frame:start_frame+split_len, :, :]

            Eta_iter_split = torch.zeros(iter_number, self.num_source, batch_size, split_len, self.num_obs).to(
                self.device)
            x_mean_dvaeumot_iter_split = torch.zeros(iter_number, self.num_source, batch_size, split_len, self.x_dim).to(
                self.device)
            x_var_dvaeumot_iter_split = torch.zeros(iter_number, self.num_source, batch_size, split_len, self.x_dim,
                                         self.x_dim).to(self.device)

            # Run the EM algorithm
            for i in range(iter_number):
                # E-W
                Eta_n_sum = torch.zeros(batch_size, split_len, self.num_obs).to(self.device)
                for n in range(self.num_source):
                    if i == 0:
                        x_mean_dvaeumot_n = x_mean_dvaeumot_init_split[n, :, :, :]
                        x_var_dvaeumot_n = x_var_dvaeumot_init_split[n, :, :, :, :]
                    else:
                        x_mean_dvaeumot_n = x_mean_dvaeumot_iter_split[i - 1, n, :, :, :]
                        x_var_dvaeumot_n = x_var_dvaeumot_iter_split[i - 1, n, :, :, :, :]                    
                    Phi = Phi_init_split
                    Phi_inv = Phi_inv_init_split

                    Eta_n = self.compute_eta_n(data_obs_split, Phi, Phi_inv, x_mean_dvaeumot_n, x_var_dvaeumot_n)
                    Eta_n_tosum = torch.clone(Eta_n)
                    Eta_n_tosum[torch.isnan(Eta_n_tosum)] = 0
                    Eta_n_sum += Eta_n_tosum
                    Eta_iter_split[i, n, :, :, :] = Eta_n
                Eta_n_sum = Eta_n_sum.expand(self.num_source, batch_size, split_len, self.num_obs)
                Eta_iter_split[i, :, :, :, :] = Eta_iter_split[i, :, :, :, :] / Eta_n_sum
                # E-S/E-Z
                with torch.no_grad():
                    for n in range(self.num_source):
                        if i == 0:
                            x_dvaeumot_im1_n = x_mean_dvaeumot_init_split[n, :, :, :]
                        else:
                            x_dvaeumot_im1_n = x_mean_dvaeumot_iter_split[i - 1, n, :, :, :]
                            
                        Phi_inv = Phi_inv_init_split
                        Eta_n = Eta_iter_split[i, n, :, :, :]
                        frame0 = x_mean_dvaeumot_init_split[n, :, 0, :]
                        x_mean_dvaeumot_n, x_var_dvaeumot_n = self.dvae_model(x_dvaeumot_im1_n, Eta_n, Phi_inv, data_obs_split, frame0, compute_loss=False)

                        x_mean_dvaeumot_iter_split[i, n, :, :, :] = x_mean_dvaeumot_n.detach()
                        x_var_dvaeumot_iter_split[i, n, :, :, :, :] = x_var_dvaeumot_n.detach()

            start_frame += split_len

        return x_mean_dvaeumot_init, x_var_dvaeumot_init, Phi_init, Phi_inv_init

    def compute_eta_n(self, o, Phi, Phi_inv, x_mean_dvaeumot_n, x_var_dvaeumot_n):
        ### Tensor dimensions
        # o: (batch_size, seq_len, num_obs, o_dim)
        # Phi: (batch_size, seq_len, num_obs, o_dim, o_dim)
        # Phi_inv: (batch_size, seq_len, num_obs, o_dim, o_dim)
        # x_mean_dvaeumot_n: (batch_size, seq_len, x_dim)
        # x_var_dvaeumot_n: (batch_size, seq_len, x_dim, x_dim)

        # Eta_n: (batch_size, seq_len, num_obs)

        seq_len = o.shape[1]
        batch_size = o.shape[0]

        Eta_n = torch.zeros(self.num_obs, batch_size, seq_len).to(self.device)

        for k in range(self.num_obs):
            Phi_k = Phi[:, :, k, :, :]
            Phi_inv_k = Phi_inv[:, :, k, :, :]
            o_k = o[:, :, k, :]

            det_Phi_k = torch.det(Phi_k)
            det_Phi_k_sqrt = torch.sqrt(det_Phi_k)

            o_ms = o_k.unsqueeze(-1) - x_mean_dvaeumot_n.unsqueeze(-1)
            o_ms_Phi = torch.matmul(o_ms.squeeze().unsqueeze(-2), Phi_inv_k)
            o_ms_Phi_sq = torch.matmul(o_ms_Phi, o_ms)
            o_ms_Phi_sq = 0.008 * o_ms_Phi_sq.squeeze()
            gaussian_exp_term = torch.exp(-0.5*o_ms_Phi_sq)

            Phi_Sigma = torch.matmul(Phi_inv_k, x_var_dvaeumot_n)
            trace = 0.008 * torch.diagonal(Phi_Sigma, dim1=-2, dim2=-1).sum(-1)
            exp_tr_term = torch.exp(-0.5 * trace)

            Beta_n = (1 / det_Phi_k_sqrt) * gaussian_exp_term * exp_tr_term + 0.0000000001
            Eta_n[k, :, :] = Beta_n / (self.num_source + 1)

        Eta_n = Eta_n.permute(1, 2, 0)

        return Eta_n

    def compute_loss_qwqs(self, eta, o, Phi, Phi_inv, x_mean_dvaeumot, x_var_dvaeumot):
        batch_size = eta.shape[1]
        seq_len = eta.shape[2]
        loss_qwqs = torch.zeros(batch_size, seq_len).to(self.device)
        for n in range(self.num_source):
            x_mean_dvaeumot_n = x_mean_dvaeumot[n, :, :, :]
            x_var_dvaeumot_n = x_var_dvaeumot[n, :, :, :, :]
            for k in range(self.num_obs):
                Phi_k = Phi[:, :, k, :, :]
                Phi_inv_k = Phi_inv[:, :, k, :, :]
                o_k = o[:, :, k, :]
                eta_kn = eta[n, :, :, k]

                det_Phi_k = torch.det(Phi_k)
                o_ms = o_k.unsqueeze(-1) - x_mean_dvaeumot_n.unsqueeze(-1)
                o_ms_Phi = torch.matmul(o_ms.squeeze().unsqueeze(-2), Phi_inv_k)
                o_ms_Phi_sq = torch.matmul(o_ms_Phi, o_ms)
                o_ms_Phi_sq = o_ms_Phi_sq.squeeze()

                Phi_Sigma = torch.matmul(Phi_inv_k, x_var_dvaeumot_n)
                trace = torch.diagonal(Phi_Sigma, dim1=-2, dim2=-1).sum(-1)

                loss_qwqs_kn = eta_kn * 0.5 * (torch.log(det_Phi_k) + o_ms_Phi_sq + trace)
                loss_qwqs_kn[torch.isnan(loss_qwqs_kn)] = 0
                loss_qwqs += loss_qwqs_kn

        loss_qwqs = torch.sum(loss_qwqs) / (self.num_source * batch_size * seq_len * self.num_obs)

        return loss_qwqs

    def compute_phi(self, o, x_mean_dvaeumot, x_var_dvaeumot, Eta, eps=0.000001):
        ### Tensor dimensions
        # o: (batch_size, seq_len, num_obs, o_dim)
        # x_mean_dvaeumot: (num_source, batch_size, seq_len, x_dim)
        # x_var_dvaeumot: (num_source, batch_size, seq_len, x_dim, x_dim)
        # Eta: (num_source, batch_size, seq_len, num_obs)

        # Phi: (batch_size, seq_len, num_obs, o_dim, o_dim)
        # Phi_inv: (batch_size, seq_len, num_obs, o_dim, o_dim)

        batch_size = o.shape[0]
        seq_len = o.shape[1]
        num_obs = o.shape[2]
        Phi = torch.zeros(num_obs, batch_size, seq_len, self.o_dim, self.o_dim).to(self.device)
        Phi_inv = torch.zeros(num_obs, batch_size, seq_len, self.o_dim, self.o_dim).to(self.device)

        for k in range(num_obs):
            Phi_k = torch.zeros(batch_size, seq_len, self.o_dim, self.o_dim).to(self.device)
            o_k = o[:, :, k, :]
            for n in range(self.num_source):
                x_mean_dvaeumot_n = x_mean_dvaeumot[n, :, :, :]
                x_var_dvaeumot_n = x_var_dvaeumot[n, :, :, :, :]
                o_ms = o_k.unsqueeze(-1) - x_mean_dvaeumot_n.unsqueeze(-1)
                o_ms_sq = torch.matmul(o_ms, o_ms.squeeze().unsqueeze(-2))

                one_source = (Eta[n, :, :, k].unsqueeze(-1) * (x_var_dvaeumot_n + o_ms_sq).view(batch_size, seq_len, self.o_dim*self.o_dim)).view(batch_size, seq_len, self.o_dim, self.o_dim)
                Phi_k += one_source

            Phi_k = Phi_k + eps * torch.eye(self.o_dim).to(self.device)
            Phi[k, :, :, :, :] = Phi_k
            try:
                for i in range(batch_size):
                    for t in range(seq_len):
                        if Phi_k[i, t].sum().isnan():
                            Phi_inv[k, i, t, :, :] = Phi_k[i, t]
                        else:
                            u = torch.cholesky(Phi_k[i, t])
                            Phi_inv[k, i, t, :, :] = torch.cholesky_inverse(u)
            except RuntimeError:
                print('Phi: {}'.format(Phi_k))
                print('Phi_inv: {}'.format(Phi_inv[k, :, :, :, :]))
                print('o_ms_sq: {}'.format(o_ms_sq))
                print('o: {}'.format(o[i, t, k, :]))
                print('x_mean_dvaeumot: {}'.format(x_mean_dvaeumot[n, i, t, :]))
                print('Eta: {}'.format(Eta[n, i, t, k]))
                print('x_var_dvaeumot: {}'.format(x_var_dvaeumot[n, i, t, :, :]))
        
        Phi = Phi.permute(1, 2, 0, 3, 4)
        Phi_inv = Phi_inv.permute(1, 2, 0, 3, 4)

        return Phi, Phi_inv












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

import torch
import torch.nn as nn
from torch.distributions.multivariate_normal import MultivariateNormal


class BaseModel(nn.Module):
    def __init__(self, cfg, device):
        super().__init__()
        self.cfg = cfg
        self.device = device

    def build_model(self):
        pass

    def reparameterization(self, mean, var, sample_mode):
        if sample_mode == 'logvar':
            std = torch.exp(0.5*var)
            eps = torch.randn_like(std)
            return eps.mul(std).add_(mean)
        elif sample_mode == 'var':
            std = torch.sqrt(var)
            eps = torch.randn_like(std)
            return eps.mul(std).add_(mean)
        elif sample_mode == 'covar':
            batch_size = mean.shape[0]
            x_dim = mean.shape[1]
            sampled_data = torch.zeros(batch_size, x_dim).to(self.device)
            for i in range(batch_size):
                m = MultivariateNormal(mean[i], var[i])
                sampled_data[i, :] = m.sample()
            return sampled_data
        elif sample_mode == 'complex':
            mean_real = mean.real
            mean_imag = mean.imag
            eps_real = torch.rand_like(mean_real)
            eps_imag = torch.rand_like(mean_imag)
            std_real = torch.sqrt(0.5 * var)
            std_imag = torch.sqrt(0.5 * var)
            
            sampled_real = eps_real.mul(std_real).add_(mean_real)
            sampled_imag = eps_imag.mul(std_imag).add_(mean_imag)
            return sampled_real + 1j * sampled_imag

    def encoder(self):
        pass

    def decoder_init(self):
        pass

    def decoder_vem(self, z, Eta, Phi_inv, o):
        pass

    def compute_var_vem_mot(self, Eta, Phi_inv, x_logvar_dec):
        ### Tensor dimensions
        # Eta: (batch_size, num_obs)
        # Phi_inv: (batch_size, num_obs, o_dim, o_dim)
        # o: (batch_size, num_obs, o_dim)
        # x_mean_dec: (batch_size, x_dim)
        # x_logvar_dec: (batch_size, x_dim)

        # x_var_vem: (batch_size, x_dim, x_dim)
        batch_size = Eta.shape[0]
        num_obs = Eta.shape[1]

        x_var_vem = torch.zeros(batch_size, self.x_dim, self.x_dim).to(self.device)
        x_var_dec_inv = 1 / x_logvar_dec.exp()
        term_obs = torch.zeros(batch_size, self.x_dim, self.x_dim).to(self.device)
        for k in range(num_obs):
            Phi_inv_k = Phi_inv.permute(1, 0, 2, 3)[k, :, :, :]
            Eta_k = Eta.permute(1, 0)[k]
            Eta_Phi = (Eta_k.unsqueeze(-1) * Phi_inv_k.view(batch_size, self.x_dim * self.x_dim)).view(batch_size, self.x_dim, self.x_dim)
            Eta_Phi[torch.isnan(Eta_Phi)] = 0
            term_obs += Eta_Phi

        term_dec = torch.zeros(batch_size, self.x_dim, self.x_dim).to(self.device)
        for i in range(batch_size):
            term_dec[i, :, :] = torch.diag(x_var_dec_inv[i, :])
        sum_terms = term_obs + term_dec
        try:
            u = torch.cholesky(sum_terms)
        except RuntimeError:
            print(sum_terms)
        for i in range(batch_size):
            x_var_vem[i, :, :] = torch.cholesky_inverse(u[i, :, :])

        return x_var_vem

    def compute_mean_vem_mot(self, Eta, Phi_inv, o, x_mean_dec, x_logvar_dec, x_var_vem):
        ### Tensor dimensions
        # Eta: (batch_size, num_obs)
        # Phi_inv: (batch_size, num_obs, o_dim, o_dim)
        # o: (batch_size, num_obs, o_dim)
        # x_mean_dec: (batch_size, x_dim)
        # x_logvar_dec: (batch_size, x_dim)
        # x_var_vem: (batch_size, x_dim, x_dim)

        # x_mean_vem: (batch_size, x_dim)
        batch_size = Eta.shape[0]
        num_obs = Eta.shape[1]

        x_var_dec_inv_diag = 1 / x_logvar_dec.exp()

        term_obs = torch.zeros(batch_size, self.x_dim).to(self.device)
        for k in range(num_obs):
            Phi_inv_k = Phi_inv.permute(1, 0, 2, 3)[k, :, :, :]
            Eta_k = Eta.permute(1, 0)[k]
            o_k = o.permute(1, 0, 2)[k, :, :]
            Phi_o = torch.matmul(Phi_inv_k, o_k.unsqueeze(-1))
            Eta_Phi_o = Eta_k.unsqueeze(-1) * Phi_o.squeeze()
            Eta_Phi_o[torch.isnan(Eta_Phi_o)] = 0
            term_obs += Eta_Phi_o
        x_var_dec_inv = torch.zeros(batch_size, self.x_dim, self.x_dim).to(self.device)
        for i in range(batch_size):
            x_var_dec_inv[i, :, :] = torch.diag(x_var_dec_inv_diag[i, :])
        term_dec = torch.matmul(x_var_dec_inv, x_mean_dec.unsqueeze(-1)).squeeze()
        sum_terms = term_obs + term_dec
        x_mean_vem = torch.matmul(x_var_vem, sum_terms.unsqueeze(-1)).squeeze()

        return x_mean_vem
    
    def compute_var_vem_scass(self, Eta, v_theta_x, v_theta_s):
        ### Tensor dimensions
        ## Inputs:
        # Eta: (batch_size, num_freq)
        # v_theta_x: (batch_size, num_freq)
        # v_theta_s: (batch_size, num_freq)

        ## Output:
        # v_phi_s: (batch_size, num_freq)

        v_theta_s_inv = 1 / v_theta_s
        v_theta_x_inv = 1 / v_theta_x
        
        v_phi_s_inv = Eta * v_theta_x_inv + v_theta_s_inv
        v_phi_s = 1 / v_phi_s_inv
        
        return v_phi_s

    def compute_mean_vem_scass(self, Eta, v_theta_x, data_mixed, v_phi_s):
        ### Tensor dimensions
        ## Inputs:
        # Eta: (batch_size, num_freq)
        # v_theta_x: (batch_size, num_freq)
        # data_mixed: (batch_size, num_feq)
        # v_phi_s: (batch_size, num_freq)

        ## Outputs:
        # mu_phi_s: (batch_size, num_freq)

        v_theta_x_inv = 1 / v_theta_x
        
        mu_phi_s = v_phi_s * Eta * v_theta_x_inv * data_mixed

        return mu_phi_s

    def vem_params_initialization(self, o):
        pass

    def prior(self):
        pass

    def forward(self, z_mean_inf_init, z_logvar_inf_init, Eta, Phi_inv, o, compute_loss):
        pass

    def get_loss(self, x, x_mean_gen, x_logvar_gen, z_mean_inf, z_logvar_inf, z_mean_prior,
                 z_logvar_prior, batch_size, seq_len, beta=1):
        pass

    def get_info(self):
        pass

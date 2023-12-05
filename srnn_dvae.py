from collections import OrderedDict
import torch
import torch.nn as nn
import numpy as np
from .base_model import BaseModel


class SRNN(BaseModel):
    def __init__(self, cfg, device):
        super(SRNN, self).__init__(cfg, device)
        # Load model parameters
        # General
        self.x_dim = self.cfg.getint('Network', 'x_dim')
        self.z_dim = self.cfg.getint('Network', 'z_dim')
        activation = self.cfg.get('Network', 'activation')
        self.schedule_sampling = self.cfg.getboolean('Training', 'schedule_sampling')
        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        else:
            raise SystemExit('Wrong activation type!')
        self.dropout_p = self.cfg.getfloat('Network', 'dropout_p')
        # Deterministic
        self.dense_x_h = [] if self.cfg.get('Network', 'dense_x_h') == 'None' else [int(i) for i in
                                                              self.cfg.get('Network', 'dense_x_h').split(',')]
        self.dim_RNN_h = self.cfg.getint('Network', 'dim_RNN_h')
        self.num_RNN_h = self.cfg.getint('Network', 'num_RNN_h')
        # Inference
        self.dense_hx_g = [] if cfg.get('Network', 'dense_hx_g') == 'None' else [int(i) for i in
                                                                        self.cfg.get('Network', 'dense_hx_g').split(',')]
        self.dim_RNN_g = self.cfg.getint('Network', 'dim_RNN_g')
        self.num_RNN_g = self.cfg.getint('Network', 'num_RNN_g')
        self.dense_gz_z = [] if self.cfg.get('Network', 'dense_gz_z') == 'None' else [int(i) for i in
                                                                        self.cfg.get('Network', 'dense_gz_z').split(',')]
        # Prior
        self.dense_hz_z = [] if self.cfg.get('Network', 'dense_hz_z') == 'None' else [int(i) for i in
                                                                        self.cfg.get('Network', 'dense_hz_z').split(',')]
        # Generation
        self.dense_hz_x = [] if self.cfg.get('Network', 'dense_hz_x') == 'None' else [int(i) for i in
                                                                        self.cfg.get('Network', 'dense_hz_x').split(',')]
        # Beta-vae
        self.beta = cfg.getfloat('Training', 'beta')

        # Build the model
        self.build_model()


    def build_model(self):
        ###############################
        ######## Deterministic ########
        ###############################

        ######## MLP Layers ########
        # 1. x_tm1 -> h_t
        self.dict_mlp_xh = OrderedDict()
        if len(self.dense_x_h) == 0:
            dim_x_h = self.x_dim
            self.dict_mlp_xh['Identity'] = nn.Identity()
        else:
            dim_x_h = self.dense_x_h[-1]
            for i in range(len(self.dense_x_h)):
                if i == 0:
                    self.dict_mlp_xh['linear_%s' % str(i)] = nn.Linear(self.x_dim, self.dense_x_h[i])
                else:
                    self.dict_mlp_xh['linear_%s' % str(i)] = nn.Linear(self.dense_x_h[i-1], self.dense_x_h[i])
                self.dict_mlp_xh['activation_%s' % str(i)] = self.activation
                self.dict_mlp_xh['dropout_%s' % str(i)] = nn.Dropout(p=self.dropout_p)

        self.mlp_x_h = nn.Sequential(self.dict_mlp_xh)

        ######## Forward Recurrent Layer ########
        # 2. h_t, forward recurrence
        self.rnn_h = nn.LSTM(dim_x_h, self.dim_RNN_h, self.num_RNN_h)
        
        self.module_deterministic_layers = [self.mlp_x_h, self.rnn_h]

        ###########################
        ######### Encoder #########
        ###########################

        ######## MLP Layers ########
        # 1. h_t x_t -> g_t
        self.dict_mlp_xhg = OrderedDict()
        if len(self.dense_hx_g) == 0:
            dim_hx_g = self.x_dim + self.dim_RNN_h
            self.dict_mlp_xhg['Identity'] = nn.Identity()
        else:
            dim_hx_g = self.dense_hx_g[-1]
            for i in range(len(self.dense_hx_g)):
                if i == 0:
                    self.dict_mlp_xhg['linear_%s' % i] = nn.Linear(self.x_dim+self.dim_RNN_h, self.dense_hx_g[i])
                else:
                    self.dict_mlp_xhg['linear_%s' % i] = nn.Linear(self.dense_hx_g[i-1], self.dense_hx_g[i])
                self.dict_mlp_xhg['activation_%s' % i] = self.activation
                self.dict_mlp_xhg['dropout_%s' % i] = nn.Dropout(p=self.dropout_p)
        self.mlp_hx_g = nn.Sequential(self.dict_mlp_xhg)

        ######## Backward Recurrent Layer ########
        # 2. g_t, backward recurrence
        self.rnn_g = nn.LSTM(dim_hx_g, self.dim_RNN_g, self.num_RNN_g)

        ######## MLP Layers ########
        # 3. g_t z_tm1 -> z_t, inference
        self.dict_mlp_gzz = OrderedDict()
        if len(self.dense_gz_z) == 0:
            dim_gz_z = self.z_dim + self.dim_RNN_g
            self.dict_mlp_gzz['Identity'] = nn.Identity()
        else:
            dim_gz_z = self.dense_gz_z[-1]
            for i in range(len(self.dense_gz_z)):
                if i == 0:
                    self.dict_mlp_gzz['linear_%s' % i] = nn.Linear(self.z_dim + self.dim_RNN_g, self.dense_gz_z[i])
                else:
                    self.dict_mlp_gzz['linear_%s' % i] = nn.Linear(self.dense_gz_z[i - 1], self.dense_gz_z[i])
                self.dict_mlp_gzz['activation_%s' % i] = self.activation
                self.dict_mlp_gzz['dropout_%s' % i] = nn.Dropout(p=self.dropout_p)

        self.mlp_gz_z = nn.Sequential(self.dict_mlp_gzz)

        self.inf_z_mean = nn.Linear(dim_gz_z, self.z_dim)
        self.inf_z_logvar = nn.Linear(dim_gz_z, self.z_dim)
        
        self.module_encoder_layers = [self.mlp_hx_g, self.rnn_g, self.mlp_gz_z, self.inf_z_mean, self.inf_z_logvar]

        ###########################
        ########## Prior ##########
        ###########################

        ######## MLP Layers ########
        # 1. h_t z_tm1 -> z_t
        self.dict_mlp_hzz = OrderedDict()
        if len(self.dense_hz_z) == 0:
            dim_hz_z = self.z_dim + self.dim_RNN_h
            self.dict_mlp_hzz['Identity'] = nn.Identity()
        else:
            dim_hz_z = self.dense_hz_z[-1]
            for i in range(len(self.dense_hz_z)):
                if i == 0:
                    self.dict_mlp_hzz['linear_%s' % i] = nn.Linear(self.z_dim + self.dim_RNN_h, self.dense_hz_z[i])
                else:
                    self.dict_mlp_hzz['linear_%s' % i] = nn.Linear(self.dense_hz_z[i-1], self.dense_hz_z[i])
                self.dict_mlp_hzz['activation_%s' % i] = self.activation
                self.dict_mlp_hzz['dropout_%s' % i] = nn.Dropout(p=self.dropout_p)
        self.mlp_hz_z = nn.Sequential(self.dict_mlp_hzz)
        self.prior_z_mean = nn.Linear(dim_hz_z, self.z_dim)
        self.prior_z_logvar = nn.Linear(dim_hz_z, self.z_dim)
        
        self.module_prior_layers = [self.mlp_hz_z, self.prior_z_logvar, self.prior_z_mean]

        ###########################
        ######### Decoder #########
        ###########################

        ######## MLP Layers ########
        # 1. h_t z_t -> x_t
        self.dict_mlp_hzx = OrderedDict()
        if len(self.dense_hz_x) == 0:
            dim_hz_x = self.z_dim + self.dim_RNN_h
            self.dict_mlp_hzx['Identity'] = nn.Identity()
        else:
            dim_hz_x = self.dense_hz_x[-1]
            for i in range(len(self.dense_hz_x)):
                if i == 0:
                    self.dict_mlp_hzx['linear_%s' % i] = nn.Linear(self.z_dim + self.dim_RNN_h, self.dense_hz_x[i])
                else:
                    self.dict_mlp_hzx['linear_%s' % i] = nn.Linear(self.dense_hz_x[i - 1], self.dense_hz_x[i])
                self.dict_mlp_hzx['activation_%s' % i] = self.activation
                self.dict_mlp_hzx['dropout_%s' % i] = nn.Dropout(p=self.dropout_p)
        self.mlp_hz_x = nn.Sequential(self.dict_mlp_hzx)
        self.gen_x = nn.Linear(dim_hz_x, self.x_dim)
        
        self.module_decoder_layers = [self.mlp_hz_x, self.gen_x]


    def deterministic_h(self, x_tm1):
        x_h = self.mlp_x_h(x_tm1)
        h, _ = self.rnn_h(x_h)
        
        return h


    def encoder(self, x, h):
        seq_len = x.shape[0]
        batch_size = x.shape[1]

        # Create variable holder and send to GPU if needed
        z_mean_inf = torch.zeros(seq_len, batch_size, self.z_dim).to(self.device)
        z_logvar_inf = torch.zeros(seq_len, batch_size, self.z_dim).to(self.device)
        z = torch.zeros(seq_len, batch_size, self.z_dim).to(self.device)
        z_t = torch.zeros(batch_size, self.z_dim).to(self.device)

        # 1. From h_t and x_t to g_t
        hx_g = torch.cat((h,x), -1)
        hx_g = self.mlp_hx_g(hx_g)
        g_inverse, _ = self.rnn_g(torch.flip(hx_g, [0]))
        g = torch.flip(g_inverse, [0])

        # 2. From g_t and z_tm1 to z_t
        # z_t here is z[t,:,:] in the last loop (or a zero tensor)
        # so it refers to z_tm1 actually
        for t in range(seq_len):
            gz_z = torch.cat((g[t, :, :], z_t), -1)
            gz_z = self.mlp_gz_z(gz_z)
            z_mean_inf[t, :, :] = self.inf_z_mean(gz_z)
            z_logvar_inf[t, :, :] = self.inf_z_logvar(gz_z)
            z_t = self.reparameterization(z_mean_inf[t, :, :], z_logvar_inf[t, :, :], 'logvar')
            z[t, :, :] = z_t

        return z, z_mean_inf, z_logvar_inf


    def prior(self, h, z_tm1):
        hz_z = torch.cat((h, z_tm1), -1)
        hz_z = self.mlp_hz_z(hz_z)
        z_mean_prior = self.prior_z_mean(hz_z)
        z_logvar_prior = self.prior_z_logvar(hz_z)

        return z_mean_prior, z_logvar_prior


    def decoder_tf(self, h, z):
        hz_x = torch.cat((h, z), -1)
        hz_x = self.mlp_hz_x(hz_x)
        x_gen_logvar = self.gen_x(hz_x)

        return x_gen_logvar
    
    
    def decoder_ss(self, h_inf, z, use_pred=1):
        seq_len = z.shape[0]
        batch_size = z.shape[1]
        
        h_gen = torch.zeros((seq_len, batch_size, self.dim_RNN_h)).to(self.device)
        x_gen_logvar = torch.zeros(seq_len, batch_size, self.x_dim).to(self.device)
        x_tm1 = torch.zeros(batch_size, self.x_dim).to(self.device)
        h_t = torch.zeros(self.num_RNN_h, batch_size, self.dim_RNN_h).to(self.device)
        c_t = torch.zeros(self.num_RNN_h, batch_size, self.dim_RNN_h).to(self.device)        
        
        for t in range(seq_len):
            x_h_tm1 = self.mlp_x_h(x_tm1.unsqueeze(0).detach())
            _, (h_t, c_t) = self.rnn_h(x_h_tm1, (h_t, c_t))
            h_t_last = h_t.view(self.num_RNN_h, 1, batch_size, self.dim_RNN_h)[-1, :, :, :]
            h_t_last = h_t_last.view(batch_size * self.num_RNN_h, self.dim_RNN_h)
            h_gen[t, :, :] = h_t_last
            z_t = z[t, :, :]
            
            # During training, randomly replace x_{t-1} by \hat{x}_{t-1}
            random_num = np.random.uniform()
            if random_num <= use_pred:
                hz_x_t = torch.cat((h_t_last, z_t), -1)
            else:
                hz_x_t = torch.cat((h_inf[t, :, :], z_t), -1)
            hzx_t = self.mlp_hz_x(hz_x_t)
            x_gen_logvar_t = self.gen_x(hzx_t)
            x_tm1 = torch.exp(x_gen_logvar_t)            
            x_gen_logvar[t, :, :] = x_gen_logvar_t
            
        return x_gen_logvar, h_gen
    

    def generate(self, seq_len, batch_size):
        x_gen_logvar = torch.zeros(seq_len, batch_size, self.x_dim).to(self.device)
        h = torch.zeros(seq_len, batch_size, self.dim_RNN_h).to(self.device)
        z_sampled = torch.zeros(seq_len, batch_size, self.z_dim).to(self.device)
        z_mean_prior = torch.zeros(seq_len, batch_size, self.z_dim).to(self.device)
        z_logvar_prior = torch.zeros(seq_len, batch_size, self.z_dim).to(self.device)
        x_t = torch.randn(batch_size, self.x_dim).to(self.device)
        z_t = torch.zeros(batch_size, self.z_dim).to(self.device)

        h_t = torch.zeros(self.num_RNN_h, batch_size, self.dim_RNN_h).to(self.device)
        c_t = torch.zeros(self.num_RNN_h, batch_size, self.dim_RNN_h).to(self.device)

        for t in range(seq_len):
            x_h_t = self.mlp_x_h(x_t).unsqueeze(0)
            _, (h_t, c_t) = self.rnn_h(x_h_t, (h_t, c_t))
            h_t_last = h_t.view(self.num_RNN_h, 1, batch_size, self.dim_RNN_h)[-1, :, :, :]
            h_t_last = h_t_last.view(batch_size * self.num_RNN_h, self.dim_RNN_h)
            h[t, :, :] = h_t_last
            z_mean_prior_t, z_logvar_prior_t = self.prior(h_t_last, z_t)
            z_t = self.reparameterization(z_mean_prior_t, z_logvar_prior_t, 'logvar')
            x_gen_logvar_t = self.decoder(h_t_last, z_t)

            x_gen_logvar[t, :, :] = x_gen_logvar_t
            z_mean_prior[t, :, :] = z_mean_prior_t
            z_logvar_prior[t, :, :] = z_logvar_prior_t
            z_sampled[t, :, :] = z_t

        x_gen = x_gen_logvar.exp().permute(1, 0, 2).squeeze()

        return x_gen

    def forward(self, x, compute_elbo_loss, use_pred=0):
        if len(x.shape) == 2:
            x = x.unsqueeze(0)
        x = x.permute(1, 0, 2).float()
        seq_len = x.shape[0]
        batch_size = x.shape[1]
        x_dim = x.shape[2]
        assert (x_dim == self.x_dim)

        x_0 = torch.zeros(1, batch_size, x_dim).to(self.device)
        x_tm1 = torch.cat((x_0, x[:-1, :, :]), 0)
        h_inf = self.deterministic_h(x_tm1)
        z, z_mean_inf, z_logvar_inf = self.encoder(x, h_inf)
        z_0 = torch.zeros(1, batch_size, self.z_dim).to(self.device)
        z_tm1 = torch.cat((z_0, z[:-1, :, :]), 0)
        if self.schedule_sampling:
            x_gen_logvar, h_gen = self.decoder_ss(h_inf, z, use_pred)
            z_mean_prior, z_logvar_prior = self.prior(h_gen, z_tm1)
        else:
            x_gen_logvar = self.decoder_tf(h_inf, z)
            z_mean_prior, z_logvar_prior = self.prior(h_inf, z_tm1)

        # Compute loss
        if compute_elbo_loss:
            loss_tot, loss_recon, loss_KLD = self.get_loss(x, x_gen_logvar, z_mean_inf, z_logvar_inf,
                                                           z_mean_prior, z_logvar_prior, batch_size, seq_len, beta=self.beta)
            self.loss = {'loss_tot': loss_tot, 'loss_recon': loss_recon, 'loss_KLD': loss_KLD}
        self.x_gen = x_gen_logvar.exp().permute(1, 0, 2).squeeze()
        self.z = z.permute(1, 0, 2).squeeze()
        self.z_mean_inf = z_mean_inf.permute(1, 0, 2).squeeze()
        self.z_logvar_inf = z_logvar_inf.permute(1, 0, 2).squeeze()
        self.z_mean_prior = z_mean_prior.permute(1, 0, 2).squeeze()
        self.z_logvar_prior = z_logvar_prior.permute(1, 0, 2).squeeze()

        return self.x_gen, self.z_mean_inf, self.z_logvar_inf, self.z_mean_prior, self.z_logvar_prior


    def get_loss(self, x, x_gen_logvar, z_mean_inf, z_logvar_inf,
                 z_mean_prior, z_logvar_prior, batch_size, seq_len, beta=1):
        loss_recon = torch.sum( x / (x_gen_logvar.exp() + 1e-10) - torch.log(x / (x_gen_logvar.exp() + 1e-10)) - 1)
        loss_KLD = -0.5 * torch.sum(z_logvar_inf - z_logvar_prior
                                    - torch.div(z_logvar_inf.exp() + (z_mean_inf - z_mean_prior).pow(2), z_logvar_prior.exp()))

        loss_recon = loss_recon / (batch_size * seq_len)
        loss_KLD = loss_KLD / (batch_size * seq_len)
        loss_tot = loss_recon + beta * loss_KLD

        return loss_tot, loss_recon, loss_KLD

    def get_info(self):
        info = []
        info.append('========== MODEL INFO ==========')
        info.append('----------- Encoder -----------')
        info.append('x_tm1 to h_t:')
        for k, v in self.dict_mlp_xh.items():
            info.append('%s : %s' % (k, str(v)))
        info.append(str(self.rnn_h))
        info.append('h_t and x_t to g_t:')
        for k, v in self.dict_mlp_xhg.items():
            info.append('%s : %s' % (k, str(v)))
        info.append(str(self.rnn_g))
        info.append('z_tm1 and g_t to z_t:')
        for k, v in self.dict_mlp_gzz.items():
            info.append('%s : %s' % (k, str(v)))
        info.append('inf z mean: ' + str(self.inf_z_mean))
        info.append('inf z logvar: ' + str(self.inf_z_logvar))

        info.append('----------- Decoder -----------')
        info.append('h_t and z_t to x_t:')
        for k, v in self.dict_mlp_hzx.items():
            info.append('%s : %s' % (k, str(v)))
        info.append('gen x: ' + str(self.gen_x))

        info.append('----------- Prior -----------')
        info.append('h_t and z_tm1 to z_t:')
        for k, v in self.dict_mlp_hzz.items():
            info.append('%s : %s' % (k, str(v)))
        info.append('prior z mean: ' + str(self.prior_z_mean))
        info.append('prior z logvar: ' + str(self.prior_z_logvar))
        info.append('\n')

        return info











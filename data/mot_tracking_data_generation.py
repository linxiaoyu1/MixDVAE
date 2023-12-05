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
import numpy as np
import pickle
import torch
from torch import nn
import pandas as pd
import itertools

def generate_data(read_path_gt, read_path_obs, read_path_social, det_type, save_dir, seq_len, slide_step, min_obs_ratio, model_ae):
    for file in os.listdir(read_path_gt):
        file_name = file.split('.')[0]
        file_number = file_name.split('_')[1]
        file_path_gt = os.path.join(read_path_gt, file)
        file_path_obs = os.path.join(read_path_obs, '{}/mathched_det_{}_{}.csv'.format(det_type, file_number, det_type))
        data_gt = pd.read_csv(file_path_gt, header=None,
                           names=['frame', 'ID', 'x', 'y', 'w', 'h', 'flag', 'object type', 'visibility ratio'])
        data_obs = pd.read_csv(file_path_obs)
        data_social = np.load(read_path_social, allow_pickle=True).item()
        data_gt['x1'] = data_gt['x']
        data_gt['x2'] = data_gt['x'] + data_gt['w']
        data_gt['y1'] = data_gt['y']
        data_gt['y2'] = data_gt['y'] - data_gt['h']
        data_obs['x1'] = data_obs['x']
        data_obs['x2'] = data_obs['x'] + data_obs['w']
        data_obs['y1'] = data_obs['y']
        data_obs['y2'] = data_obs['y'] - data_obs['h']

        time_index_list = []
        frame_start = data_gt['frame'].min()
        total_seq_len = data_gt['frame'].max()

        while (frame_start + seq_len) < total_seq_len:
            frame_end = frame_start + seq_len
            time_index_list.append(np.arange(frame_start, frame_end))
            frame_start += slide_step

        for t in range(len(time_index_list)):
            start_frame = time_index_list[t][0]
            end_frame = time_index_list[t][-1] + 1
            data_split_gt = data_gt[(data_gt['frame'] >= start_frame) & (data_gt['frame'] < end_frame)]
            data_split_obs = data_obs[(data_obs['frame'] >= start_frame) & (data_obs['frame'] < end_frame)]
            count_gt = data_split_gt[data_split_gt['flag'] == 1].groupby(by='ID').count()['frame']
            count_obs = data_split_obs.groupby(by='ID').count()['frame']
            valid_id_brut = np.array(count_gt[count_gt == seq_len].index)
            valid_id_minobs = np.array(count_obs[count_obs >= int(seq_len*min_obs_ratio)].index)
            valid_id_trueobs_path = 'MOT17_all/obs_validid/{}/mathchedid_det_{}_{}.txt'.format(det_type, file_number, det_type)
            valid_id_trueobs = np.loadtxt(valid_id_trueobs_path)
            valid_id = []
            for idx in valid_id_brut:
                if (idx in valid_id_trueobs) & (idx in valid_id_minobs):
                    valid_id.append(idx)
            valid_id = np.array(valid_id)

            if len(valid_id) < 3:
                continue

            id_combins = []
            for i in range(20):
                idx = np.random.choice(valid_id, 3, replace=False)
                idx.sort()
                id_combins.append(idx)
            id_combins = np.unique(id_combins, axis=0)
            # id_combins = list(itertools.combinations(valid_id, 3))
            np.sort(id_combins)
            for tracking_ids in id_combins:
                valid_data_s1_gt = data_split_gt[data_split_gt['ID'] == tracking_ids[0]][
                    ['frame', 'x1', 'y1', 'x2', 'y2', 'w', 'h']]
                valid_data_s2_gt = data_split_gt[data_split_gt['ID'] == tracking_ids[1]][
                    ['frame', 'x1', 'y1', 'x2', 'y2', 'w', 'h']]
                valid_data_s3_gt = data_split_gt[data_split_gt['ID'] == tracking_ids[2]][
                    ['frame', 'x1', 'y1', 'x2', 'y2', 'w', 'h']]
                valid_data_s1_obs = data_split_obs[data_split_obs['ID'] == tracking_ids[0]][
                    ['frame', 'x1', 'y1', 'x2', 'y2', 'w', 'h']]
                valid_data_s2_obs = data_split_obs[data_split_obs['ID'] == tracking_ids[1]][
                    ['frame', 'x1', 'y1', 'x2', 'y2', 'w', 'h']]
                valid_data_s3_obs = data_split_obs[data_split_obs['ID'] == tracking_ids[2]][
                    ['frame', 'x1', 'y1', 'x2', 'y2', 'w', 'h']]

                valid_data_s1_obs = valid_data_s1_obs.rename(
                    columns={"x1": "x1_obs", "y1": "y1_obs", "x2": "x2_obs", "y2": "y2_obs", "w": "w_obs",
                             "h": "h_obs"})
                valid_data_s2_obs = valid_data_s2_obs.rename(
                    columns={"x1": "x1_obs", "y1": "y1_obs", "x2": "x2_obs", "y2": "y2_obs", "w": "w_obs",
                             "h": "h_obs"})
                valid_data_s3_obs = valid_data_s3_obs.rename(
                    columns={"x1": "x1_obs", "y1": "y1_obs", "x2": "x2_obs", "y2": "y2_obs", "w": "w_obs",
                             "h": "h_obs"})
                joint_valid_data_s1 = valid_data_s1_gt.set_index('frame').join(valid_data_s1_obs.set_index('frame'))
                joint_valid_data_s2 = valid_data_s2_gt.set_index('frame').join(valid_data_s2_obs.set_index('frame'))
                joint_valid_data_s3 = valid_data_s3_gt.set_index('frame').join(valid_data_s3_obs.set_index('frame'))

                if np.isnan(joint_valid_data_s1['x1_obs'].iloc[:5].values.sum()):
                    continue
                if np.isnan(joint_valid_data_s2['x1_obs'].iloc[:5].values.sum()):
                    continue
                if np.isnan(joint_valid_data_s3['x1_obs'].iloc[:5].values.sum()):
                    continue

                dir = "MOT17-{}-FRCNN".format(file_number)
                social_rep_s1 = generate_social_representation(tracking_ids[0], data_social, dir, seq_len, start_frame, end_frame, model_ae)
                social_rep_s2 = generate_social_representation(tracking_ids[1], data_social, dir, seq_len, start_frame, end_frame, model_ae)
                social_rep_s3 = generate_social_representation(tracking_ids[2], data_social, dir, seq_len, start_frame, end_frame, model_ae)

                s1_xywh = np.expand_dims(
                    np.asarray(joint_valid_data_s1[['x1', 'y1', 'w', 'h']],
                               dtype='float64'), axis=1)
                s2_xywh = np.expand_dims(
                    np.asarray(joint_valid_data_s2[['x1', 'y1', 'w', 'h']],
                               dtype='float64'), axis=1)
                s3_xywh = np.expand_dims(
                    np.asarray(joint_valid_data_s3[['x1', 'y1', 'w', 'h']],
                               dtype='float64'), axis=1)
                gt_sequences_xywh = np.concatenate((s1_xywh, s2_xywh, s3_xywh), axis=1)

                s1_x1x2 = np.expand_dims(
                    np.asarray(joint_valid_data_s1[['x1', 'y1', 'x2', 'y2']],
                               dtype='float64'), axis=1)
                s2_x1x2 = np.expand_dims(
                    np.asarray(joint_valid_data_s2[['x1', 'y1', 'x2', 'y2']],
                               dtype='float64'), axis=1)
                s3_x1x2 = np.expand_dims(
                    np.asarray(joint_valid_data_s3[['x1', 'y1', 'x2', 'y2']],
                               dtype='float64'), axis=1)
                gt_sequences_x1x2 = np.concatenate((s1_x1x2, s2_x1x2, s3_x1x2), axis=1)

                o1_xywh = np.expand_dims(
                    np.asarray(joint_valid_data_s1[['x1_obs', 'y1_obs', 'w_obs', 'h_obs']],
                               dtype='float64'), axis=1)
                o2_xywh = np.expand_dims(
                    np.asarray(joint_valid_data_s2[['x1_obs', 'y1_obs', 'w_obs', 'h_obs']],
                               dtype='float64'), axis=1)
                o3_xywh = np.expand_dims(
                    np.asarray(joint_valid_data_s3[['x1_obs', 'y1_obs', 'w_obs', 'h_obs']],
                               dtype='float64'), axis=1)
                det_sequences_xywh = np.concatenate((o1_xywh, o2_xywh, o3_xywh), axis=1)

                o1_x1x2 = np.expand_dims(
                    np.asarray(joint_valid_data_s1[['x1_obs', 'y1_obs', 'x2_obs', 'y2_obs']],
                               dtype='float64'), axis=1)
                o2_x1x2 = np.expand_dims(
                    np.asarray(joint_valid_data_s2[['x1_obs', 'y1_obs', 'x2_obs', 'y2_obs']],
                               dtype='float64'), axis=1)
                o3_x1x2 = np.expand_dims(
                    np.asarray(joint_valid_data_s3[['x1_obs', 'y1_obs', 'x2_obs', 'y2_obs']],
                               dtype='float64'), axis=1)
                det_sequences_x1x2 = np.concatenate((o1_x1x2, o2_x1x2, o3_x1x2), axis=1)

                social = np.concatenate((social_rep_s1.to('cpu'), social_rep_s2.to('cpu'), social_rep_s3.to('cpu')), axis=0)

                one_data_xywh = {'gt': gt_sequences_xywh, 'det': det_sequences_xywh, 'social': social}
                one_data_x1x2 = {'gt': gt_sequences_x1x2, 'det': det_sequences_x1x2}

                save_path_xywh = os.path.join(save_dir, 'xywh/{}'.format(det_type))
                save_path_x1x2 = os.path.join(save_dir, 'x1x2/{}'.format(det_type))
                save_path_data_xywh = os.path.join(save_path_xywh, '{}_p{}_p{}_p{}_t{}.pkl'.format(file_number, tracking_ids[0],
                                                                            tracking_ids[1], tracking_ids[2], t))
                save_path_data_x1x2 = os.path.join(save_path_x1x2, '{}_p{}_p{}_p{}_t{}.pkl'.format(file_number,
                                                                                          tracking_ids[0],
                                                                                          tracking_ids[1],
                                                                                          tracking_ids[2], t))

                if not(os.path.isdir(save_path_xywh)):
                    os.makedirs(save_path_xywh)
                if not(os.path.isdir(save_path_x1x2)):
                    os.makedirs(save_path_x1x2)                    
                with open(save_path_data_xywh, 'wb') as file:
                    pickle.dump(one_data_xywh, file)
                with open(save_path_data_x1x2, 'wb') as file:
                    pickle.dump(one_data_x1x2, file)


def generate_social_representation(track_id, data_social, dir, seq_len, start_frame, end_frame, model_ae):
    social_dict = {}
    social_ids = data_social[dir]['tracklets'][track_id]['social_ids']
    social_ids = np.unique(social_ids)
    for sid in social_ids:
        start_s = data_social[dir]['tracklets'][sid]['start']
        end_s = data_social[dir]['tracklets'][sid]['end']
        SEQ = np.zeros((seq_len, 4))
        START, END = -1, -1

        if start_s == start_frame and end_frame == end_s:
            SEQ = data_social[dir]['tracklets'][sid]['sequence'][0:-1]
            START = start_s
            END = end_s
        elif start_frame < end_s <= end_frame:
            if start_s >= start_frame:
                SEQ = data_social[dir]['tracklets'][sid]['sequence'][0:-1]
                START = start_s
                END = end_s
            else:
                idx = start_frame - start_s
                SEQ = data_social[dir]['tracklets'][sid]['sequence'][idx:-1]
                START = start_frame
                END = end_s
        elif end_s > start_frame and start_s < end_frame:
            if start_s >= start_frame:
                idx_end = end_frame - start_s
                SEQ = data_social[dir]['tracklets'][sid]['sequence'][:idx_end]
                START = start_s
                END = end_frame
            elif start_s < start_frame:
                idx_end = end_frame - start_s
                idx_start = start_frame - start_s
                SEQ = data_social[dir]['tracklets'][sid]['sequence'][idx_start:idx_end]
                START = start_frame
                END = end_frame
        if START != -1 and END != -1 and START != END:
            social_dict[sid] = {'seq': SEQ, 'start': START, 'end': END}

    current_batch_social = torch.zeros(len(social_dict.keys()), seq_len, 4)
    idx = 0
    for key, value in social_dict.items():
        start_social = int(value['start'])
        end_social = int(value['end'])
        if start_social > end_social:
            pass
        else:
            if start_social == start_frame and end_social == end_frame:
                current_batch_social[idx] = torch.from_numpy(value['seq'])

            if start_social > start_frame and end_social < end_frame:
                current_batch_social[idx, :(start_social - start_frame)] = torch.from_numpy(value['seq'][0, :])
                current_batch_social[idx, (start_social - start_frame):-(end_frame - end_social)] = torch.from_numpy(
                    value['seq'][:, :])
                current_batch_social[idx, -(end_frame - end_social):] = torch.from_numpy(value['seq'][-1, :])
            elif end_social < end_frame and start_social == start_frame:
                current_batch_social[idx, -(end_frame - end_social):] = torch.from_numpy(value['seq'][-1, :])
                current_batch_social[idx, :-(end_frame - end_social)] = torch.from_numpy(value['seq'][:, :])
            elif end_social == end_frame and start_social > start_frame:
                current_batch_social[idx, :(start_social - start_frame)] = torch.from_numpy(value['seq'][0, :])
                current_batch_social[idx, (start_social - start_frame):] = torch.from_numpy(value['seq'][:, :])
            else:
                current_batch_social[idx] = torch.from_numpy(value['seq'])

        idx += 1
    social_vel = torch.zeros(current_batch_social.shape)
    social_vel[:, 1:, :] = current_batch_social[:, 1:, :] - current_batch_social[:, :-1, :]
    with torch.no_grad():
        social_vel = social_vel.float().cuda()
        try:
            h = model_ae.inference(social_vel)
            h = torch.max(h, dim=0)[0].unsqueeze(0)
        except:
            h = torch.zeros(1, seq_len, 256).float().cuda()

    return h


class motion_ae(nn.Module):

    def __init__(self, hidden_state):

        super(motion_ae, self).__init__()
        self.hidden_size = hidden_state
        self.encoder_fc = nn.Linear(4, self.hidden_size // 2)
        self.encoder = nn.GRU(self.hidden_size // 2, self.hidden_size, num_layers=1, batch_first=True)
        self.decoder = nn.GRU(4, self.hidden_size, num_layers=1, batch_first=True)
        self.decoder_fc = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size // 4),
            nn.ReLU(),
            nn.Linear(self.hidden_size // 4, 4)
        )

    def forward(self, observation, tf):
        observation_enc = nn.ReLU()(self.encoder_fc(observation))
        _, encoder_h = self.encoder(observation_enc)

        T = observation.shape[1]
        mask = np.random.uniform(size=T - 1) < tf

        reconstructed = []
        init_motion = observation[:, 0:1, :]

        x, h = self.decoder(init_motion, encoder_h)
        x = self.decoder_fc(x)
        x = x + init_motion

        reconstructed.append(x)

        for t in range(1, T):
            if mask[t - 1]:
                x_t, h = self.decoder(observation[:, t:t + 1, :], h)
            else:
                x_t, h = self.decoder(x, h)

            x_t = self.decoder_fc(x_t)
            x = x_t + x

            reconstructed.append(x)

        return torch.cat(reconstructed, dim=1)

    def inference(self, observation):
        observation = self.encoder_fc(observation)
        hiddens = []
        h = None
        for t in range(observation.shape[1]):
            _, h = self.encoder(observation[:, t:t+1, :], h)
            hiddens.append(h)
        hiddens = torch.cat(hiddens, dim=0)
        hiddens = hiddens.permute(1, 0, 2)
        return hiddens

if __name__ == '__main__':
    read_path_gt = 'MOT17_all/gt_orig_data'
    read_path_obs = 'MOT17_all/matched_det_data'
    save_dir = 'MOT17-test/short_seq'
    read_path_social = 'ArTIST_data/postp_train_mot17.npy'
    seq_len = 60
    min_obs_ratio = 0.65
    slide_step = 40
    det_type = 'sdp'
    model_ae = motion_ae(256).cuda()
    model_ae.load_state_dict(torch.load("../models/ArTIST/ae/ae_8.pth"))
    model_ae.eval()
    generate_data(read_path_gt, read_path_obs, read_path_social, det_type, save_dir, seq_len, slide_step, min_obs_ratio, model_ae)

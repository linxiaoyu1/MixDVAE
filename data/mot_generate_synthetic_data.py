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
import numpy as np

def power_function(start, params, frame_sequence):
    # params = [a_2, a_1], start = a_0
    out = params[0] * (frame_sequence**2) + params[1] * frame_sequence + start
    return out

def sinus_function(start, params, frame_sequence):
    # params = [omega, phi_0]
    out = start/np.sin(params[1]) * np.sin(params[0]*frame_sequence + params[1])
    return out

def generate_synthetic_trajectory(total_seq_len, num_segment, function_type_list, function_type_proba, std_params, start_point):
    trajectory_x = []
    # The split points are between [3, total_seq_len], and the minimum subsequence length length is 3 frames.
    segment_pts = np.random.choice(np.arange(3, total_seq_len, 3), num_segment, replace=False)
    segment_pts = np.concatenate([segment_pts, np.array([60])])
    segment_pts = np.sort(segment_pts)
    for i in range(len(segment_pts)):
        if i == 0:
            true_subseq_len = segment_pts[i]
            sub_seq_end = true_subseq_len
            sub_frame_sequence = np.arange(sub_seq_end)
            start_x = start_point
        else:
            true_subseq_len = segment_pts[i]-segment_pts[i-1]
            sub_seq_end = true_subseq_len
            sub_frame_sequence = np.arange(sub_seq_end)
            start_x = trajectory_x[i-1][-1]
        function_type_x = np.random.choice(function_type_list, p=function_type_proba)
        if function_type_x == 'static':
            params = [0, 0]
            subtraj_x = power_function(start_x, params, sub_frame_sequence)
        elif function_type_x == 'line':
            v_x = std_params['vx'] * np.random.randn()
            params = [0, v_x]
            subtraj_x = power_function(start_x, params, sub_frame_sequence)
        elif function_type_x == 'parabol':
            v_x = std_params['vx'] * np.random.randn()
            a_x = std_params['ax'] * np.random.randn()
            params = [a_x, v_x]
            subtraj_x = power_function(start_x, params, sub_frame_sequence)
        elif function_type_x == 'sinus':
            omega = std_params['omega'] * np.random.randn()
            phi = std_params['phi'] * np.random.randn()
            params = [omega, phi]
            subtraj_x = sinus_function(start_x, params, sub_frame_sequence)
        if len(subtraj_x) < true_subseq_len:
            subtraj_x = np.concatenate([subtraj_x, subtraj_x[-1]*np.ones(true_subseq_len - len(subtraj_x))])
        trajectory_x.append(subtraj_x)
    trajectory_x = np.concatenate(trajectory_x)

    return trajectory_x

if __name__ == '__main__':
    save_dir = 'data/synthetic_trajectories/'
    # Define parameters, distribution parameters estimated from MOT17 public detection.
    max_training_set_len = 20000
    max_validation_set_len = 5000

    total_seq_len = 60
    n_split_max = 3
    std_params_x = {'vx': 0.00125, 'ax': 0.000037, 'omega': 0.003, 'phi': 3.2}
    std_params_y = {'vx': 0.0006, 'ax': 0.00002, 'omega': 0.001, 'phi': 3.2}
    std_params_w = {'vx': 0.00032, 'ax':0.00001}
    params_w = [-3.8, 0.7]
    params_ratio = [5.26, 0.77]
    function_type_list_xy = ['static', 'line', 'parabol', 'sinus']
    function_type_proba_xy = [0.15, 0.3, 0.3, 0.25]
    function_type_w = ['static', 'line', 'parabol']
    function_type_proba_w = [0.25, 0.6, 0.15]

    # Generate trajectories
    # Training data
    for i in range(max_training_set_len):
        num_segment_x = np.random.choice(np.arange(n_split_max))
        num_segment_y = np.random.choice(np.arange(n_split_max))
        num_segment_w = 0

        start_point_x = np.random.rand()
        start_point_y = np.random.rand()
        start_point_w = np.random.lognormal(params_w[0], params_w[1])

        trajectory_x = generate_synthetic_trajectory(total_seq_len, num_segment_x, function_type_list_xy,
                                                     function_type_proba_xy, std_params_x,
                                                     start_point_x).reshape(-1, 1)
        trajectory_y = generate_synthetic_trajectory(total_seq_len, num_segment_y, function_type_list_xy,
                                                     function_type_proba_xy, std_params_y,
                                                     start_point_y).reshape(-1, 1)

        w = generate_synthetic_trajectory(total_seq_len, num_segment_w, function_type_w, function_type_proba_w,
                                          std_params_w, start_point_w).reshape(-1, 1)
        h_w_ratio = params_ratio[1] * np.random.randn() + params_ratio[0]
        h = w * h_w_ratio
        w_min = np.min(w)
        one_data = np.concatenate((trajectory_x, trajectory_y, trajectory_x + w, trajectory_y - h), axis=1)

        x1_max = np.max(one_data[:, 0])
        y1_max = np.max(one_data[:, 1])
        x2_max = np.max(one_data[:, 2])
        y2_max = np.max(one_data[:, 3])

        x1_min = np.min(one_data[:, 0])
        y1_min = np.min(one_data[:, 1])
        x2_min = np.min(one_data[:, 2])
        y2_min = np.min(one_data[:, 3])

        # Ignore the data that going out of the image
        if (x1_min<-0.5) or (x2_min<0) or (x1_max>1) or (x2_max>0.8) or (y1_min<0) or (y2_min<-1.5) or (y1_max>1.5) or (y2_max>1) or (w_min<0):
            continue

        save_path = os.path.join(save_dir, 'train_data')
        if not(os.path.isdir(save_path)):
            os.makedirs(save_path)
        save_path_onedata = os.path.join(save_path, 'data_{}.pkl'.format(i))
        with open(save_path_onedata, 'wb') as file:
            pickle.dump(one_data, file)

    # Validation data
    for i in range(max_validation_set_len):
        num_segment_x = np.random.choice(np.arange(n_split_max))
        num_segment_y = np.random.choice(np.arange(n_split_max))
        num_segment_w = 0

        start_point_x = np.random.rand()
        start_point_y = np.random.rand()
        start_point_w = np.random.lognormal(params_w[0], params_w[1])

        trajectory_x = generate_synthetic_trajectory(total_seq_len, num_segment_x, function_type_list_xy,
                                                     function_type_proba_xy, std_params_x,
                                                     start_point_x).reshape(-1, 1)
        trajectory_y = generate_synthetic_trajectory(total_seq_len, num_segment_y, function_type_list_xy,
                                                     function_type_proba_xy, std_params_y,
                                                     start_point_y).reshape(-1, 1)

        w = generate_synthetic_trajectory(total_seq_len, num_segment_w, function_type_w, function_type_proba_w,
                                          std_params_w, start_point_w).reshape(-1, 1)
        h_w_ratio = params_ratio[1] * np.random.randn() + params_ratio[0]
        h = w * h_w_ratio
        w_min = np.min(w)
        one_data = np.concatenate((trajectory_x, trajectory_y, trajectory_x + w, trajectory_y - h), axis=1)

        x1_max = np.max(one_data[:, 0])
        y1_max = np.max(one_data[:, 1])
        x2_max = np.max(one_data[:, 2])
        y2_max = np.max(one_data[:, 3])

        x1_min = np.min(one_data[:, 0])
        y1_min = np.min(one_data[:, 1])
        x2_min = np.min(one_data[:, 2])
        y2_min = np.min(one_data[:, 3])

        # Ignore the data that going out of the image
        if (x1_min < -0.5) or (x2_min < 0) or (x1_max > 1) or (x2_max > 0.8) or (y1_min < 0) or (y2_min < -1.5) or (
                y1_max > 1.5) or (y2_max > 1) or (w_min < 0):
            continue
        save_path = os.path.join(save_dir, 'val_data')
        if not(os.path.isdir(save_path)):
            os.makedirs(save_path)
        save_path_onedata = os.path.join(save_path, 'data_{}.pkl'.format(i))
        with open(save_path_onedata, 'wb') as file:
            pickle.dump(one_data, file)

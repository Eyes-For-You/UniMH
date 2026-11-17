"""Teacher data loading and preprocessing; datasets are not distributed."""
import os
import pickle
import glob
import random
import numpy as np
import math
from einops import rearrange
import torch
import matplotlib.pyplot as plt
from utils.normalization import normalize_min_max

def seq_collate_giga(batch):
    index, past_traj, fut_traj, past_traj_orig, fut_traj_orig, traj_vel, past_len, fut_len, past_mask, fut_mask = zip(*batch)
    indexes = torch.stack(index, dim=0)
    pre_motion_3D = torch.stack(past_traj, dim=0)
    fut_motion_3D = torch.stack(fut_traj, dim=0)
    pre_motion_3D_orig = torch.stack(past_traj_orig, dim=0)
    fut_motion_3D_orig = torch.stack(fut_traj_orig, dim=0)
    fut_traj_vel = torch.stack(traj_vel, dim=0)
    past_len = torch.stack(past_len, dim=0)
    fut_len = torch.stack(fut_len, dim=0)
    past_mask = torch.stack(past_mask, dim=0)
    fut_mask = torch.stack(fut_mask, dim=0)
    batch_size = torch.tensor(pre_motion_3D.shape[0])
    data = {'indexes': indexes, 'batch_size': batch_size, 'past_traj': pre_motion_3D, 'fut_traj': fut_motion_3D, 'past_traj_original_scale': pre_motion_3D_orig, 'fut_traj_original_scale': fut_motion_3D_orig, 'fut_traj_vel': fut_traj_vel, 'past_len': past_len, 'fut_len': fut_len, 'past_mask': past_mask, 'fut_mask': fut_mask}
    return data

def rotate_traj(past_rel, future_rel, past_abs, agents=2, rotate_time_frame=0, subset='eth'):
    past_rel = rearrange(past_rel, 'b a p d -> (b a) p d')
    past_abs = rearrange(past_abs, 'b a p d -> (b a) p d')
    future_rel = rearrange(future_rel, 'b a f d -> (b a) f d')
    past_diff = past_rel[:, rotate_time_frame]
    past_theta = torch.atan(torch.div(past_diff[:, 1], past_diff[:, 0] + 1e-05))
    past_theta = torch.where(past_diff[:, 0] < 0, past_theta + math.pi, past_theta)
    rotate_matrix = torch.zeros((past_theta.size(0), 2, 2)).to(past_theta.device)
    rotate_matrix[:, 0, 0] = torch.cos(past_theta)
    rotate_matrix[:, 0, 1] = torch.sin(past_theta)
    rotate_matrix[:, 1, 0] = -torch.sin(past_theta)
    rotate_matrix[:, 1, 1] = torch.cos(past_theta)
    past_after = torch.matmul(rotate_matrix, past_rel.transpose(1, 2)).transpose(1, 2)
    future_after = torch.matmul(rotate_matrix, future_rel.transpose(1, 2)).transpose(1, 2)
    past_abs_after = torch.matmul(rotate_matrix, past_abs.transpose(1, 2)).transpose(1, 2)
    past_after = rearrange(past_after, '(b a) p d -> b a p d', a=agents)
    future_after = rearrange(future_after, '(b a) f d -> b a f d', a=agents)
    past_abs_after = rearrange(past_abs_after, '(b a) p d -> b a p d', a=agents)
    return (past_after, future_after, past_abs_after)

class GigaDataset(object):

    def __init__(self, cfg, training=True, data_dir=None, subset=None, rotate_time_frame=0, type='original'):
        if type == 'LED':
            data_file_path = os.path.join(data_dir, type, '{:s}_data_{:s}.npy'.format(subset, 'train' if training else 'test'))
            num_file_path = os.path.join(data_dir, type, '{:s}_num_{:s}.npy'.format(subset, 'train' if training else 'test'))
            all_data = np.load(data_file_path)
            all_num = np.load(num_file_path)
            self.all_data = torch.Tensor(all_data)
            self.all_num = torch.Tensor(all_num)
        elif type == 'original':
            data_file_path = os.path.join(data_dir, type, subset, '{:s}_{:s}.pkl'.format(subset, 'train' if training else 'test'))
            all_data = pickle.load(open(data_file_path, 'rb'))
            all_data = all_data['traj']
            self.all_data = torch.Tensor(all_data)
            self.all_data = self.all_data[:, None, :, :]
        elif type == 'TUTR':
            data_file_path = os.path.join(data_dir, type, subset, '{:s}_{:s}.pkl'.format(subset, 'train' if training else 'test'))
            all_data = pickle.load(open(data_file_path, 'rb'))
            all_data = all_data['traj']
            self.all_data = torch.Tensor(all_data)
            self.all_data = self.all_data[:, None, :, :]
        else:
            raise ValueError('Invalid type')
        self.cfg = cfg
        self.training = training
        self.rotate_time_frame = rotate_time_frame
        self.past_frames_list = cfg.past_frames if isinstance(cfg.past_frames, list) else [cfg.past_frames]
        self.future_frames_list = cfg.future_frames if isinstance(cfg.future_frames, list) else [cfg.future_frames]
        self.max_past_frames = cfg.max_past_frames
        self.max_future_frames = cfg.max_future_frames
        self.test_past_frames = getattr(cfg, 'test_past_frames', self.max_past_frames)
        self.test_future_frames = getattr(cfg, 'test_future_frames', self.max_future_frames)
        if not training:
            print(f'[GigaDataset] training={training}, test_past_frames={self.test_past_frames}, test_future_frames={self.test_future_frames}')
        cfg.agents = self.all_data.shape[1]
        cfg.MODEL.CONTEXT_ENCODER.AGENTS = cfg.agents
        last_past_frames = self.max_past_frames
        past_traj_abs = self.all_data[:, :, :last_past_frames]
        initial_pos = past_traj_abs[:, :, -1:]
        past_traj_rel = (past_traj_abs - initial_pos).contiguous()
        fut_traj_full = (self.all_data[:, :, last_past_frames:] - initial_pos).contiguous()
        actual_fut_len = fut_traj_full.shape[2]
        if actual_fut_len >= self.max_future_frames:
            fut_traj = fut_traj_full[:, :, :self.max_future_frames]
        else:
            pad_size = self.max_future_frames - actual_fut_len
            fut_traj = torch.cat([fut_traj_full, torch.zeros(*fut_traj_full.shape[:2], pad_size, fut_traj_full.shape[-1])], dim=2)
        if cfg.rotate:
            past_traj_rel, fut_traj, past_traj_abs = rotate_traj(past_traj_rel, fut_traj, past_traj_abs, cfg.agents, rotate_time_frame, subset)
        past_traj_vel = torch.cat((past_traj_rel[:, :, 1:] - past_traj_rel[:, :, :-1], torch.zeros_like(past_traj_rel[:, :, -1:])), dim=2)
        past_traj = torch.cat((past_traj_abs, past_traj_rel, past_traj_vel), dim=-1)
        self.fut_traj_vel = torch.cat((fut_traj[:, :, 1:] - fut_traj[:, :, :-1], torch.zeros_like(fut_traj[:, :, -1:])), dim=2)
        self.rotate_aug = cfg.rotate_aug and training
        if training:
            cfg.fut_traj_max = fut_traj.max()
            cfg.fut_traj_min = fut_traj.min()
            cfg.past_traj_max = past_traj.max()
            cfg.past_traj_min = past_traj.min()
        self.past_traj_original_scale = past_traj
        self.fut_traj_original_scale = fut_traj
        if cfg.data_norm == 'min_max':
            self.past_traj = normalize_min_max(past_traj, cfg.past_traj_min, cfg.past_traj_max, -1, 1).contiguous()
            self.fut_traj = normalize_min_max(fut_traj, cfg.fut_traj_min, cfg.fut_traj_max, -1, 1).contiguous()
        elif cfg.data_norm == 'original':
            self.past_traj = past_traj
            self.fut_traj = fut_traj
        'load distillation target'
        self.eval_mode = False

    def set_eval_mode(self, eval_mode=True):
        """Set evaluation mode to use fixed test lengths instead of random sampling."""
        self.eval_mode = eval_mode

    def __len__(self):
        return self.all_data.shape[0]

    def _sample_lengths(self):
        """Randomly sample past_len and fut_len from configured lists."""
        if self.training and (not self.eval_mode):
            past_len = random.choice(self.past_frames_list)
            fut_len = random.choice(self.future_frames_list)
        else:
            past_len = self.test_past_frames
            fut_len = self.test_future_frames
        return (past_len, fut_len)

    def _apply_past_mask(self, past_traj, past_len):
        """
        Zero-mask the past frames that are not used.
        past_traj: [A, max_past_frames, D]
        past_len: int, number of frames to keep (from the end)
        Returns: masked past_traj, past_mask [max_past_frames] (True=valid)
        """
        max_p = self.max_past_frames
        mask_frames = max_p - past_len
        past_traj = past_traj.clone()
        past_traj[:, :mask_frames] = 0
        past_mask = torch.zeros(max_p, dtype=torch.bool)
        past_mask[mask_frames:] = True
        return (past_traj, past_mask)

    def _apply_fut_mask(self, fut_traj, fut_len):
        """
        Zero-mask the future frames that are not used.
        fut_traj: [A, max_future_frames, D]
        fut_len: int, number of frames to keep (from the start)
        Returns: masked fut_traj, fut_mask [max_future_frames] (True=valid)
        """
        max_f = self.max_future_frames
        fut_traj = fut_traj.clone()
        fut_traj[:, fut_len:] = 0
        fut_mask = torch.zeros(max_f, dtype=torch.bool)
        fut_mask[:fut_len] = True
        return (fut_traj, fut_mask)

    def __getitem__(self, item):
        past_len, fut_len = self._sample_lengths()
        past_traj_norm_scale = self.past_traj[item]
        fut_traj_norm_scale = self.fut_traj[item]
        past_traj_original_scale = self.past_traj_original_scale[item]
        fut_traj_original_scale = self.fut_traj_original_scale[item]
        fut_traj_vel = self.fut_traj_vel[item]
        if self.rotate_aug:
            A = past_traj_norm_scale.size(0)
            rot_angle = torch.rand(A) * 2 * math.pi
            past_traj_abs_o, past_traj_rel_o, past_traj_vel_o = past_traj_original_scale.chunk(3, dim=-1)
            rotate_matrix = torch.zeros((rot_angle.size(0), 2, 2)).to(past_traj_norm_scale.device)
            rotate_matrix[:, 0, 0] = torch.cos(rot_angle)
            rotate_matrix[:, 0, 1] = torch.sin(rot_angle)
            rotate_matrix[:, 1, 0] = -torch.sin(rot_angle)
            rotate_matrix[:, 1, 1] = torch.cos(rot_angle)
            past_traj_abs_o_rot = torch.matmul(rotate_matrix, past_traj_abs_o.transpose(1, 2)).transpose(1, 2)
            past_traj_rel_o_rot = torch.matmul(rotate_matrix, past_traj_rel_o.transpose(1, 2)).transpose(1, 2)
            fut_traj_o_rot = torch.matmul(rotate_matrix, fut_traj_original_scale.transpose(1, 2)).transpose(1, 2)
            past_traj_vel_o_rot = torch.cat((past_traj_rel_o_rot[:, :, 1:] - past_traj_rel_o_rot[:, :, :-1], torch.zeros_like(past_traj_rel_o_rot[:, :, -1:])), dim=2)
            past_traj_rot = torch.cat((past_traj_abs_o_rot, past_traj_rel_o_rot, past_traj_vel_o_rot), dim=-1)
            fut_traj_vel_o = torch.cat((fut_traj_o_rot[:, :, 1:] - fut_traj_o_rot[:, :, :-1], torch.zeros_like(fut_traj_o_rot[:, :, -1:])), dim=2)
            past_traj_norm_scale = normalize_min_max(past_traj_rot, self.cfg.past_traj_min, self.cfg.past_traj_max, -1, 1).contiguous()
            fut_traj_norm_scale = normalize_min_max(fut_traj_o_rot, self.cfg.fut_traj_min, self.cfg.fut_traj_max, -1, 1).contiguous()
            past_traj_original_scale = past_traj_rot
            fut_traj_original_scale = fut_traj_o_rot
            fut_traj_vel = fut_traj_vel_o
        past_traj_norm_scale, past_mask = self._apply_past_mask(past_traj_norm_scale, past_len)
        past_traj_original_scale, _ = self._apply_past_mask(past_traj_original_scale, past_len)
        fut_traj_norm_scale, fut_mask = self._apply_fut_mask(fut_traj_norm_scale, fut_len)
        fut_traj_original_scale, _ = self._apply_fut_mask(fut_traj_original_scale, fut_len)
        fut_traj_vel, _ = self._apply_fut_mask(fut_traj_vel, fut_len)
        out = [torch.Tensor([item]).to(torch.int32), past_traj_norm_scale, fut_traj_norm_scale, past_traj_original_scale, fut_traj_original_scale, fut_traj_vel, torch.tensor(past_len, dtype=torch.long), torch.tensor(fut_len, dtype=torch.long), past_mask, fut_mask]
        return out

class GigaDatasetSocialGAN:
    """
    For the purpose of saving pickle files for train/test data for Giga dataset
    """

    def __init__(self, data_dir, obs_len=8, pred_len=12, skip=1, min_ped=1, delim='\t', subset='eth'):
        super(GigaDatasetSocialGAN, self).__init__()
        self.max_peds_in_frame = 0
        self.data_dir = data_dir
        self.obs_len = obs_len
        self.pred_len = pred_len
        self.skip = skip
        self.seq_len = self.obs_len + self.pred_len
        self.delim = delim
        self.cur_frame_no = 0
        all_files_train = sorted(os.listdir(self.data_dir + '/train/'))
        all_files_test = sorted(os.listdir(self.data_dir + '/test/'))
        all_files_train = [os.path.join(self.data_dir + '/train/', path) for path in all_files_train]
        all_files_test = [os.path.join(self.data_dir + '/test/', path) for path in all_files_test]
        data_dict = {'train': all_files_train, 'test': all_files_test}
        for type, all_files in data_dict.items():
            num_peds_in_seq = []
            seq_list = []
            frame_list = []
            total_peds_checked = 0
            skipped_span_mismatch = 0
            skipped_incomplete_frames = 0
            peds_kept = 0
            for path in all_files:
                data = self.read_file(path, delim)
                frames = np.unique(data[:, 0]).tolist()
                frame_data = []
                for frame in frames:
                    frame_data.append(data[frame == data[:, 0], :])
                num_sequences = int(math.ceil((len(frames) - self.seq_len + 1) / skip))
                for idx in range(0, num_sequences * self.skip + 1, skip):
                    curr_seq_data = np.concatenate(frame_data[idx:idx + self.seq_len], axis=0)
                    peds_in_curr_seq = np.unique(curr_seq_data[:, 1])
                    self.max_peds_in_frame = max(self.max_peds_in_frame, len(peds_in_curr_seq))
                    curr_seq = np.zeros((len(peds_in_curr_seq), 2, self.seq_len))
                    num_peds_considered = 0
                    for _, ped_id in enumerate(peds_in_curr_seq):
                        total_peds_checked += 1
                        curr_ped_seq = curr_seq_data[curr_seq_data[:, 1] == ped_id, :]
                        curr_ped_seq = np.around(curr_ped_seq, decimals=4)
                        pad_front = frames.index(curr_ped_seq[0, 0]) - idx
                        pad_end = frames.index(curr_ped_seq[-1, 0]) - idx + 1
                        if pad_end - pad_front != self.seq_len:
                            skipped_span_mismatch += 1
                            continue
                        if len(curr_ped_seq) != self.seq_len:
                            skipped_incomplete_frames += 1
                            continue
                        curr_ped_seq = np.transpose(curr_ped_seq[:, 2:])
                        _idx = num_peds_considered
                        curr_seq[_idx, :, pad_front:pad_end] = curr_ped_seq
                        num_peds_considered += 1
                        peds_kept += 1
                    if num_peds_considered > min_ped:
                        num_peds_in_seq.append(num_peds_considered)
                        seq_list.append(curr_seq[:num_peds_considered])
                        frame_list.append(frames[idx])
            total_skipped = skipped_span_mismatch + skipped_incomplete_frames
            print(f"\n{'=' * 60}")
            print(f'Data Statistics for {type.upper()} set:')
            print(f"{'=' * 60}")
            print(f'Total pedestrian trajectories checked: {total_peds_checked}')
            print(f'  ├─ Kept (complete): {peds_kept} ({peds_kept / total_peds_checked * 100:.2f}%)')
            print(f'  └─ Skipped (incomplete): {total_skipped} ({total_skipped / total_peds_checked * 100:.2f}%)')
            print(f'     ├─ Span mismatch (first-to-last < {self.seq_len}): {skipped_span_mismatch} ({skipped_span_mismatch / total_peds_checked * 100:.2f}%)')
            print(f'     └─ Missing intermediate frames: {skipped_incomplete_frames} ({skipped_incomplete_frames / total_peds_checked * 100:.2f}%)')
            print(f"{'=' * 60}\n")
            self.num_seq = len(seq_list)
            self.trajs = np.concatenate(seq_list, axis=0).transpose(0, 2, 1)
            cum_start_idx = [0] + np.cumsum(num_peds_in_seq).tolist()
            self.seq_start_end = [(start, end) for start, end in zip(cum_start_idx, cum_start_idx[1:])]
            self.frame_list = np.array(frame_list, dtype=np.int32)
            traj_data = {'traj': self.trajs, 'frame_list': self.frame_list, 'num_peds_in_seq': num_peds_in_seq, 'seq_start_end': self.seq_start_end}
            pickle.dump(traj_data, open(self.data_dir + '/{:s}.pkl'.format(type), 'wb'))

    def read_file(self, _path, delim):
        delim = delim if delim else self.delim
        data = []
        with open(_path, 'r') as f:
            for line in f:
                line = line.strip().split(delim)
                line = [float(i) for i in line]
                data.append(line)
        return np.asarray(data)

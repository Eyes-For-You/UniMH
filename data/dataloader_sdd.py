"""Teacher data loading and preprocessing; datasets are not distributed."""
import os
import random
from glob import glob
from einops import rearrange
import math
import matplotlib.pyplot as plt
import torch.nn.functional as F
import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm
import pickle
from utils.normalization import normalize_min_max
from torch.nn.utils.rnn import pad_sequence

def rotate_traj(past_rel, future_rel, past_abs, rotate_time_frame=0):
    """
    @params past_rel: [N, A, P, 2]
    @params future_rel: [N, A, F, 2]
    @params past_abs: [N, A, P, 2]
    @params rotate_time_frame: int
    """
    A = past_rel.size(1)
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
    past_after = rearrange(past_after, '(b a) p d -> b a p d', a=A)
    future_after = rearrange(future_after, '(b a) f d -> b a f d', a=A)
    past_abs_after = rearrange(past_abs_after, '(b a) p d -> b a p d', a=A)
    return (past_after, future_after, past_abs_after)

def seq_collate_sdd(batch):
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

class SDDDataset(Dataset):

    def __init__(self, cfg, data_dir, training=True, overfit=False, rotate_time_frame=0):
        super(SDDDataset, self).__init__()
        'init'
        self.cfg = cfg
        dataset_file = os.path.join(data_dir, 'original/sdd_train.pkl') if training else os.path.join(data_dir, 'original/sdd_test.pkl')
        if overfit:
            dataset_file = os.path.join(data_dir, 'original/sdd_train.pkl')
        self.training = training
        self.overfit = overfit
        self.rotate_time_frame = rotate_time_frame
        self.past_frames_list = cfg.past_frames if isinstance(cfg.past_frames, list) else [cfg.past_frames]
        self.future_frames_list = cfg.future_frames if isinstance(cfg.future_frames, list) else [cfg.future_frames]
        self.max_past_frames = cfg.max_past_frames
        self.max_future_frames = cfg.max_future_frames
        self.test_past_frames = getattr(cfg, 'test_past_frames', self.max_past_frames)
        self.test_future_frames = getattr(cfg, 'test_future_frames', self.max_future_frames)
        self.max_agents_per_scene = 0
        'load the data in the original scale'
        all_data = pickle.load(open(dataset_file, 'rb'))
        print('Mode: {:s}, {:d} sequences'.format('train' if training else 'test', len(all_data)))
        'process the data'
        cfg.MODEL.CONTEXT_ENCODER.AGENTS = cfg.agents
        last_past_frames = self.max_past_frames
        past_traj_abs = torch.from_numpy(np.stack([scene[0] for scene in all_data], axis=0)).unsqueeze(1)
        initial_pos = past_traj_abs[:, :, -1:, :]
        past_traj_rel = (past_traj_abs - initial_pos).contiguous()
        fut_traj_abs = torch.from_numpy(np.stack([scene[1] for scene in all_data], axis=0)).unsqueeze(1)
        fut_traj_rel = (fut_traj_abs - initial_pos).contiguous()
        actual_fut_len = fut_traj_rel.shape[2]
        if actual_fut_len >= self.max_future_frames:
            fut_traj_rel = fut_traj_rel[:, :, :self.max_future_frames]
        else:
            pad_size = self.max_future_frames - actual_fut_len
            fut_traj_rel = torch.cat([fut_traj_rel, torch.zeros(*fut_traj_rel.shape[:2], pad_size, fut_traj_rel.shape[-1])], dim=2)
        if cfg.rotate:
            past_traj_rel, fut_traj_rel, past_traj_abs = rotate_traj(past_traj_rel, fut_traj_rel, past_traj_abs, rotate_time_frame)
        past_traj_vel = torch.cat((past_traj_rel[:, :, 1:] - past_traj_rel[:, :, :-1], torch.zeros_like(past_traj_rel[:, :, -1:])), dim=2)
        past_traj = torch.cat((past_traj_abs, past_traj_rel, past_traj_vel), dim=-1)
        self.fut_traj_vel = torch.cat((fut_traj_rel[:, :, 1:] - fut_traj_rel[:, :, :-1], torch.zeros_like(fut_traj_rel[:, :, -1:])), dim=2)
        self.rotate_aug = cfg.rotate_aug and training
        if training:
            cfg.fut_traj_max = fut_traj_rel.max()
            cfg.fut_traj_min = fut_traj_rel.min()
            cfg.past_traj_max = past_traj.max()
            cfg.past_traj_min = past_traj.min()
        self.past_traj_original_scale = past_traj
        self.fut_traj_original_scale = fut_traj_rel
        self.past_traj = normalize_min_max(past_traj, cfg.past_traj_min, cfg.past_traj_max, -1, 1).contiguous()
        self.fut_traj = normalize_min_max(fut_traj_rel, cfg.fut_traj_min, cfg.fut_traj_max, -1, 1).contiguous()
        'load distillation target'
        self.eval_mode = False

    def set_eval_mode(self, eval_mode=True):
        """Set evaluation mode to use fixed test lengths instead of random sampling."""
        self.eval_mode = eval_mode

    def __len__(self):
        return len(self.past_traj)

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
        past_traj_norm_scale, past_mask = self._apply_past_mask(past_traj_norm_scale, past_len)
        past_traj_original_scale, _ = self._apply_past_mask(past_traj_original_scale, past_len)
        fut_traj_norm_scale, fut_mask = self._apply_fut_mask(fut_traj_norm_scale, fut_len)
        fut_traj_original_scale, _ = self._apply_fut_mask(fut_traj_original_scale, fut_len)
        fut_traj_vel, _ = self._apply_fut_mask(fut_traj_vel, fut_len)
        out = [torch.Tensor([item]).to(torch.int32), past_traj_norm_scale, fut_traj_norm_scale, past_traj_original_scale, fut_traj_original_scale, fut_traj_vel, torch.tensor(past_len, dtype=torch.long), torch.tensor(fut_len, dtype=torch.long), past_mask, fut_mask]
        return out

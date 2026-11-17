"""Dataset-free software smoke check; generated values are not experiment results."""
import argparse
import json
import logging
import os
from pathlib import Path

os.environ.setdefault('MPLBACKEND', 'Agg')
import torch
import yaml
from easydict import EasyDict

from models.backbone_eth_ucy import ETHMotionTransformer
from models.flow_matching import FlowMatcher
from trainer.denoising_model_trainers import build_optimizer, build_scheduler


def check(dataset):
    torch.manual_seed(42)
    cfg = EasyDict(yaml.safe_load((Path(__file__).parent/f'cfg/{dataset}/cor_fm.yml').read_text()))
    cfg.update(device='cpu', agents=1, rotate=True, rotate_time_frame=6, rotate_aug=False,
               data_norm='min_max', fm_wrapper='direct', fm_rew_sqrt=False,
               fm_in_scaling=True, tied_noise=True, drop_method='emb',
               drop_logi_k=20.0, drop_logi_m=0.5, logit_norm_mean=-0.5,
               logit_norm_std=1.5, solver='euler', sampling_steps=2, sample_log_steps=False,
               LOSS_NN_MODE='agent', LOSS_REG_REDUCTION='sum', LOSS_REG_SQUARED=False,
               LOSS_VELOCITY=False, fut_traj_min=-1.0, fut_traj_max=1.0)
    cfg.MODEL.CONTEXT_ENCODER.AGENTS = 1
    cfg.MODEL.USE_PRE_NORM = False
    logger = logging.getLogger('software-selfcheck')
    logger.addHandler(logging.NullHandler())
    model = FlowMatcher(cfg, ETHMotionTransformer(cfg.MODEL, logger, cfg), logger)
    b, p, f = 2, cfg.max_past_frames, cfg.max_future_frames
    past = torch.randn(b, 1, p, 6)
    future = torch.randn(b, 1, f, 2)
    mask = torch.ones(b, f, dtype=torch.bool)
    mask[0, f//2:] = False
    future *= mask[:, None, :, None]
    batch = dict(batch_size=torch.tensor(b), past_traj=past,
                 past_traj_original_scale=past, fut_traj=future,
                 fut_traj_original_scale=future, fut_traj_vel=torch.zeros_like(future),
                 past_mask=torch.ones(b, p, dtype=torch.bool), fut_mask=mask,
                 past_len=torch.full((b,),p), fut_len=mask.sum(-1))
    optimizer = build_optimizer(model, cfg.OPTIMIZATION)
    scheduler = build_scheduler(optimizer, cfg.OPTIMIZATION, 2)
    model.train()
    loss = model(batch, {'cur_epoch':0})[0]
    assert torch.isfinite(loss)
    loss.backward()
    assert all(torch.isfinite(t.grad).all() for t in model.parameters() if t.grad is not None)
    assert any(t.grad is not None and torch.count_nonzero(t.grad) for t in model.vqvae.parameters())
    optimizer.step()
    scheduler.step()
    model.eval()
    calls=[]
    handle=model.vqvae.register_forward_hook(lambda *args: calls.append(1))
    with torch.no_grad():
        predicted = model.sample(batch, cfg.denoising_head_preds)[0]
    handle.remove()
    assert predicted.shape == (b,20,1,2*f) and torch.isfinite(predicted).all()
    assert not calls, 'PTR should not execute during teacher sampling.'
    return dict(dataset=dataset, train_forward=True, finite_backward=True,
                optimizer_step=True, sampler_shape=list(predicted.shape),
                ptr_skipped_during_inference=True)


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', choices=['all','eth_ucy','sdd','giga'], default='all')
    args=parser.parse_args()
    torch.set_num_threads(2)
    selected=['eth_ucy','sdd','giga'] if args.dataset=='all' else [args.dataset]
    print(json.dumps({'status':'pass', 'synthetic_software_check_only':True,
                      'checks':[check(d) for d in selected]}, indent=2))

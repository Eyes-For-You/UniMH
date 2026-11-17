"""Evaluate a fixed teacher checkpoint at multiple input/output lengths."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import os
from pathlib import Path

os.environ.setdefault('MPLBACKEND', 'Agg')
import numpy as np
import torch
from torch.utils.data import DataLoader
import yaml
from easydict import EasyDict

from models.backbone_eth_ucy import ETHMotionTransformer
from models.flow_matching import FlowMatcher
from utils.normalization import unnormalize_min_max
from utils.utils import set_random_seed


def lengths(value):
    if ':' in value:
        start, end = map(int, value.split(':'))
        return list(range(start, end+1))
    return [int(v) for v in value.split(',')]


def dataset(cfg, data_dir, training):
    if cfg.dataset == 'eth_ucy':
        from data.dataloader_eth_ucy import ETHDataset, seq_collate_eth
        return ETHDataset(cfg, training=training, data_dir=str(data_dir),
                          subset=cfg.subset, rotate_time_frame=cfg.rotate_time_frame,
                          type='original'), seq_collate_eth
    if cfg.dataset == 'sdd':
        from data.dataloader_sdd import SDDDataset, seq_collate_sdd
        return SDDDataset(cfg, training=training, data_dir=str(data_dir),
                          rotate_time_frame=cfg.rotate_time_frame), seq_collate_sdd
    if cfg.dataset == 'giga':
        from data.dataloader_giga import GigaDataset, seq_collate_giga
        return GigaDataset(cfg, training=training, data_dir=str(data_dir),
                           subset='giga', rotate_time_frame=cfg.rotate_time_frame,
                           type='TUTR'), seq_collate_giga
    raise ValueError('Supported datasets: eth_ucy, sdd, giga')


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cfg', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--data-dir', type=Path, required=True,
                        help='Dataset root containing original/ or TUTR/.')
    parser.add_argument('--subset', default=None)
    parser.add_argument('--histories', type=lengths, default=None,
                        help='Comma-separated lengths, e.g. 4,6,8.')
    parser.add_argument('--horizons', type=lengths, default=None,
                        help='Comma-separated lengths or inclusive range, e.g. 30:90.')
    parser.add_argument('--steps', type=int, default=100)
    parser.add_argument('--solver', choices=['euler', 'lin_poly'], default='lin_poly')
    parser.add_argument('--poly-p', type=int, default=5)
    parser.add_argument('--poly-long-step', type=int, default=1000)
    parser.add_argument('--batch-size', type=int, default=100)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', choices=['cuda', 'cpu'], default='cuda')
    parser.add_argument('--output', type=Path, default=Path('outputs/evaluation'))
    parser.add_argument('--check-only', action='store_true',
                        help='Validate data, masks and checkpoint on CPU without sampling.')
    args = parser.parse_args()
    torch.set_num_threads(2)
    cfg = EasyDict(yaml.safe_load(args.cfg.read_text()))
    for key, value in dict(rotate=True, rotate_time_frame=6, rotate_aug=False,
                           data_norm='min_max', fm_wrapper='direct',
                           fm_rew_sqrt=False, fm_in_scaling=True, tied_noise=True).items():
        cfg.setdefault(key, value)
    cfg.update(device='cpu' if args.check_only else args.device, use_wandb=False,
               sample_log_steps=False, sampling_steps=args.steps, solver=args.solver,
               lin_poly_p=args.poly_p, lin_poly_long_step=args.poly_long_step)
    cfg.subset = args.subset or cfg.get('subset', cfg.dataset)
    if cfg.dataset == 'eth_ucy' and cfg.subset not in ['eth', 'hotel', 'univ', 'zara1', 'zara2']:
        parser.error('ETH-UCY requires --subset eth|hotel|univ|zara1|zara2.')
    histories = sorted(set(args.histories or [cfg.test_past_frames]))
    horizons = sorted(set(args.horizons or cfg.get('eval_frames', [cfg.max_future_frames])))
    if not histories or not all(1 <= p <= cfg.max_past_frames for p in histories):
        parser.error('History lengths must lie within the trained input capacity.')
    if not horizons or not all(1 <= f <= cfg.max_future_frames for f in horizons):
        parser.error('Prediction endpoints must lie within the output capacity.')
    if args.batch_size < 1 or args.steps < 1:
        parser.error('Batch size and sampling steps must be positive.')
    if cfg.device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA is unavailable. Use --device cpu or --check-only.')
    if cfg.data_norm != 'min_max':
        raise ValueError('This evaluator implements the paper min-max preprocessing.')

    # Recover normalization only from the training split, then evaluate test data.
    train, _ = dataset(cfg, args.data_dir, True)
    train_count = len(train)
    del train
    cfg.test_past_frames, cfg.test_future_frames = histories[0], cfg.max_future_frames
    test, collate = dataset(cfg, args.data_dir, False)
    logger = logging.getLogger('teacher-evaluation')
    logger.addHandler(logging.NullHandler())
    model = FlowMatcher(cfg, ETHMotionTransformer(cfg.MODEL, logger, cfg), logger)
    state = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
    model.load_state_dict(state['model'], strict=True)
    del state
    for past in histories:
        test.test_past_frames = past
        example = test[0]
        assert int(example[6]) == past and int(example[7]) == cfg.max_future_frames
        assert int(example[8].sum()) == past
        assert not torch.count_nonzero(example[3][:, :cfg.max_past_frames-past])
    if args.check_only:
        print('PASS: teacher checkpoint, normalization, data and latest-history masks.')
        return

    args.output.mkdir(parents=True, exist_ok=True)
    if (args.output/'metrics.csv').exists():
        raise FileExistsError('Choose a new output directory; metrics.csv already exists.')
    model.to(cfg.device).eval()
    records = []
    for past in histories:
        test.test_past_frames = past
        set_random_seed(args.seed)
        loader = DataLoader(test, batch_size=args.batch_size, shuffle=False,
                            num_workers=0, collate_fn=collate)
        errors = []
        for index, batch in enumerate(loader):
            batch = {k:v.to(cfg.device) for k,v in batch.items()}
            n, k = int(batch['batch_size']), cfg.denoising_head_preds
            normalized = model.sample(batch, num_trajs=k)[0]
            predicted = normalized.reshape(n, k, cfg.agents, cfg.max_future_frames, 2).permute(0, 2, 1, 3, 4)
            predicted = unnormalize_min_max(predicted, cfg.fut_traj_min, cfg.fut_traj_max, -1, 1)
            distances = torch.linalg.vector_norm(predicted-batch['fut_traj_original_scale'][:, :, None], dim=-1)
            batch_errors = torch.stack([
                torch.stack([distances[..., :f].mean(-1).min(-1).values,
                             distances[..., f-1].min(-1).values], dim=-1)
                for f in horizons], dim=-2).reshape(-1, len(horizons), 2)
            assert torch.isfinite(batch_errors).all()
            errors.append(batch_errors.cpu().numpy())
            print(f'history={past}: batch {index+1}/{len(loader)}', flush=True)
        metrics = np.concatenate(errors).mean(axis=0, dtype=np.float64)
        for future, (ade, fde) in zip(horizons, metrics):
            records.append(dict(Tp=past, Tf=future, ADE=float(ade), FDE=float(fde)))
    with (args.output/'metrics.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=['Tp', 'Tf', 'ADE', 'FDE'])
        writer.writeheader()
        writer.writerows(records)
    metadata = dict(dataset=cfg.dataset, subset=cfg.subset, test_samples=len(test),
                    train_samples_for_normalization=train_count,
                    histories=histories, horizons=horizons,
                    checkpoint_sha256=sha256(args.checkpoint), config_sha256=sha256(args.cfg),
                    weights='model (online teacher)', proposals=cfg.denoising_head_preds,
                    seed=args.seed, batch_size=args.batch_size, sampling_steps=args.steps,
                    solver=args.solver, poly_p=args.poly_p, poly_long_step=args.poly_long_step,
                    device=cfg.device, precision='float32',
                    protocol='Latest history frames, fixed forecast origin; maximum-length forecast prefixes. ADE/FDE independently minimized per agent, then averaged.',
                    coordinate_units='Original preprocessed dataset coordinates.')
    (args.output/'protocol.json').write_text(json.dumps(metadata, indent=2)+'\n')
    print(f'Wrote {len(records)} ADE/FDE pairs to {args.output}/metrics.csv')


if __name__ == '__main__':
    main()

"""Evaluate UniMH with native metrics and optional sample export."""
import os
import torch
import argparse
from glob import glob

from tensorboardX import SummaryWriter


from utils.config import Config
from utils.utils import set_random_seed, log_config_to_file

from trainer.denoising_model_trainers import Trainer
from fm import build_data_loader, build_network


def parse_config(argv=None):
    """
    Parse the command line arguments and return the configuration options.
    """

    selector = argparse.ArgumentParser(add_help=False)
    selector.add_argument('--dataset', choices=['eth_ucy', 'sdd', 'giga'], default='eth_ucy')
    selected, _ = selector.parse_known_args(argv)
    dataset = selected.dataset
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', required=True, choices=['eth_ucy', 'sdd', 'giga'])

    # Basic configuration
    parser.add_argument('--ckpt_path', type=str, default=None, help='Path to the checkpoint to load the model from.')
    parser.add_argument('--cfg', default='auto', type=str, help="Config file path")
    parser.add_argument('--exp', default='', type=str, help='Experiment description for each run, name of the saving folder.')
    parser.add_argument('--save_samples', default=False, action='store_true', help='Save the samples during evaluation.')
    parser.add_argument('--eval_on_train', default=False, action='store_true', help='Evaluate the model on the training set.')

    # Data configuration
    if dataset != 'sdd':
        parser.add_argument('--data_source', default='TUTR' if dataset == 'giga' else 'original', type=str, help='Data source for the experiment. Either be original or preprocessed ones from LED.')
    parser.add_argument('--batch_size', default=None, type=int, help='Override the batch size in the config file.')
    parser.add_argument('--data_dir', type=str, default=f'./data/{dataset}', help='Directory where the data is stored.')
    parser.add_argument('--n_train', type=int, default=32500, help='Number training scenes used.')
    parser.add_argument('--n_test', type=int, default=12500, help='Number testing scenes used.')
    parser.add_argument('--data_norm', default='min_max', choices=['min_max', 'sqrt'], help='Normalization method for the data.')
    if dataset != 'sdd':
        parser.add_argument('--subset', type=str, required=True, choices=['eth', 'hotel', 'univ', 'zara1', 'zara2', 'giga'] if dataset == 'giga' else ['eth', 'hotel', 'univ', 'zara1', 'zara2'], help='Trajectory subset to run experiment')
    parser.add_argument('--rotate', default=False, action='store_true', help="Whether to rotate the trajectories in the dataset")
    parser.add_argument('--rotate_time_frame', type=int, default=0, help='Index of time frames to rotate the trajectories.')

    # Reproducibility configuration
    parser.add_argument('--fix_random_seed', action='store_true', default=False, help='fix random seed for reproducibility')
    parser.add_argument('--seed', type=int, default=42, help='Set the random seed to split the testing set for training evaluation.')

    ### FM parameters ###
    parser.add_argument('--sampling_steps', type=int, default=10, help='Number of sampling timesteps for the FlowMatcher.')
    parser.add_argument('--solver', type=str, default='euler', choices=['euler', 'lin_poly'], help='Solver for the FlowMatcher.')
    parser.add_argument('--lin_poly_p', type=int, default=2, help='Degree of the polynomial in the linear stage.')
    parser.add_argument('--lin_poly_long_step', type=int, default=1000, help='Number of steps to mimic slope in the linear stage.')
    ### FM parameters ###


    return parser.parse_args(argv)


def init_basics(args):
    """
    Init the basic configurations for the experiment.
    """

    """Load the config file"""
    result_dir = os.path.abspath(os.path.join(args.ckpt_path, '../../'))
    if args.cfg == 'auto':
        yml_ls = glob(result_dir+'/*.yml')
        assert len(yml_ls) >= 1, 'At least one config file should be found in the directory.'
        yml_path = [f for f in yml_ls if '_updated.yml' in os.path.basename(f)][0]
        args.cfg = yml_path
    cfg = Config(args.cfg, f'{args.exp}', train_mode=False)
    if cfg.dataset != args.dataset:
        raise ValueError('Configuration dataset does not match --dataset.')

    tag = '_'


    ### Update FM parameters ###
    def _update_fm_params(args, cfg, tag):
        if cfg.denoising_method == 'fm':
            cfg.sampling_steps = args.sampling_steps
            cfg.solver = args.solver

            if args.solver == 'euler':
                solver_tag_ = args.solver
            elif args.solver == 'lin_poly':
                cfg.lin_poly_p = args.lin_poly_p
                cfg.lin_poly_long_step = args.lin_poly_long_step
                solver_tag_ = f'lin_poly_p{args.lin_poly_p}_long{args.lin_poly_long_step}'
            
            fm_tag_ = f'FM_S{cfg.sampling_steps}_{solver_tag_}'
            tag += fm_tag_
            cfg.solver_tag = fm_tag_

        return cfg, tag

    cfg, tag = _update_fm_params(args, cfg, tag)


    ### Update data configuration ###
    def _update_data_params(args, cfg, tag):    

        if args.n_train != 32500:
            tag += f'_subset{args.n_train}'

        return cfg, tag

    cfg, tag = _update_data_params(args, cfg, tag)


    def _update_optimization_params(args, cfg, tag):
        if args.batch_size is not None:
            # override the batch size
            cfg.train_batch_size = args.batch_size
            cfg.test_batch_size = args.batch_size
        return cfg, tag

    cfg, tag = _update_optimization_params(args, cfg, tag)
    
    ### voila, create the saving directory ###
    tag += '_train_set' if args.eval_on_train else '_test_set'
    tag = tag.replace('__', '_')
    cfg.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logger = cfg.create_dirs(tag_suffix=tag)


    """fix random seed"""
    if args.fix_random_seed:
        set_random_seed(args.seed)


    """set up tensorboard and text log"""
    tb_dir = os.path.abspath(os.path.join(cfg.log_dir, '../tb_eval'))
    os.makedirs(tb_dir, exist_ok=True)
    tb_log = SummaryWriter(log_dir=tb_dir)

    
    """print the config file"""
    log_config_to_file(cfg.yml_dict, logger=logger)
    return cfg, logger, tb_log


def main():
    """
    Evaluate a saved checkpoint.
    """

    """Init everything"""
    args = parse_config()

    cfg, logger, tb_log = init_basics(args)

    train_loader, test_loader = build_data_loader(cfg, args, shuffle_train=False)

    denoiser = build_network(cfg, args, logger)

    """Train or evaluate the model"""
    trainer = Trainer(
        cfg,
        denoiser, 
        train_loader, 
        test_loader, 
        tb_log=tb_log,
        logger=logger,
        gradient_accumulate_every=1,
        ema_decay = 0.995,
        ema_update_every = 1,
        save_samples=args.save_samples,
        ) ### grid search

    trainer.test(mode='best', eval_on_train=args.eval_on_train) 


if __name__ == "__main__":
    main()

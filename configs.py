# -*- coding: utf-8 -*-

import argparse
from pathlib import Path
import pprint

project_dir = Path(__file__).resolve().parent

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def resolve_path(path):
    path = Path(path)
    return path if path.is_absolute() else project_dir / path

class Config(object):
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)

        self.dataset_dir = resolve_path(self.dataset_dir)
        self.semantic_dataset = resolve_path(self.semantic_dataset)
        self.splits_file = resolve_path(self.splits_file)
        self.set_dataset_dir(self.video_type)

    def set_dataset_dir(self, video_type='summe'):
        split_name = f'split{self.split_index}'
        self.log_dir = project_dir / video_type / 'logs' / split_name
        self.score_dir = project_dir / video_type / 'results' / split_name
        self.save_dir = project_dir / video_type / 'models' / split_name

    def __repr__(self):
        return 'Configurations\n' + pprint.pformat(self.__dict__)

def get_config(parse=True, **optional_kwargs):
    parser = argparse.ArgumentParser()

    # General
    parser.add_argument('--mode', type=str, default='train')
    parser.add_argument('--split_index', type=int, default=0)
    parser.add_argument('--verbose', type=str2bool, default=True)
    parser.add_argument('--video_type', type=str, default='summe')

    # Input paths & dimensions
    parser.add_argument('--dataset_dir', type=str, required=True)
    parser.add_argument('--semantic_dataset', type=str, required=True)
    parser.add_argument('--splits_file', type=str, required=True)
    # parser.add_argument('--baseline', type=str, required=True)
    parser.add_argument('--input_size', type=int, default=1024)
    parser.add_argument('--semantic_size', type=int, default=768)
    parser.add_argument('--hidden_size', type=int, default=512)

    # Mamba Hyperparameters
    parser.add_argument('--d_state', type=int, default=16)
    parser.add_argument('--dropout', type=float, default=0.1)

    # Summary Constraints
    parser.add_argument('--summary_rate', type=float, default=0.15)

    # Loss Weights
    parser.add_argument('--lambda_recon', type=float, default=1.0)
    parser.add_argument('--lambda_sparse', type=float, default=5.0)
    parser.add_argument('--lambda_div', type=float, default=0.1)
    parser.add_argument('--lambda_smooth', type=float, default=0.05)

    # Training & Warmup
    parser.add_argument('--warmup_epochs', type=int, default=5)
    parser.add_argument('--n_epochs', type=int, default=50)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=1e-5)
    parser.add_argument('--clip', type=float, default=5.0)
    parser.add_argument('--lr_scorer', type=float, default=1e-3)
    

    if parse:
        kwargs = parser.parse_args()
    else:
        kwargs = parser.parse_known_args()[0]

    kwargs = vars(kwargs)
    kwargs.update(optional_kwargs)

    return Config(**kwargs)
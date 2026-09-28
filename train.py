# -*- coding: utf-8 -*-

from configs import get_config
from Solver import Solver
from data_loader import get_loader

if __name__ == '__main__':
    config = get_config(mode='train')

    print("========== CONFIGURATION ==========")
    print(config)
    print("===================================\n")

    train_loader = get_loader(
        mode='train',
        split_index=config.split_index,
        cnn_filename=config.dataset_dir,
        semantic_filename=config.semantic_dataset,
        splits_filename=config.splits_file
    )

    test_loader = get_loader(
        mode='test',
        split_index=config.split_index,
        cnn_filename=config.dataset_dir,
        semantic_filename=config.semantic_dataset,
        splits_filename=config.splits_file
    )

    solver = Solver(config, train_loader, test_loader)
    solver.build()
    solver.train()
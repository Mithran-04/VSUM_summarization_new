from __future__ import print_function
import os.path as osp
import argparse
import h5py
import numpy as np

from utils import write_json


parser = argparse.ArgumentParser("Code to create 5-fold splits in json form")

parser.add_argument(
    '-d',
    '--dataset',
    type=str,
    required=True,
    help="path to h5 dataset (required)"
)

parser.add_argument(
    '--save-dir',
    type=str,
    default='datasets',
    help="path to save output json file (default: 'datasets/')"
)

parser.add_argument(
    '--save-name',
    type=str,
    default='splits',
    help="name to save as, excluding extension (default: 'splits')"
)

parser.add_argument(
    '--num-splits',
    type=int,
    default=5,
    help="number of folds (default: 5)"
)

parser.add_argument(
    '--seed',
    type=int,
    default=42,
    help="random seed for reproducible folds"
)

args = parser.parse_args()


def create():
    print("==========\nArgs:{}\n==========".format(args))

    print("Loading dataset from {}".format(args.dataset))

    dataset = h5py.File(args.dataset, 'r')

    # Convert HDF5 keys to a normal Python list
    keys = list(dataset.keys())

    num_videos = len(keys)

    print("Total videos: {}".format(num_videos))

    if num_videos % args.num_splits != 0:
        raise ValueError(
            "Number of videos ({}) must be divisible by "
            "number of folds ({}) for this 5-fold split.".format(
                num_videos,
                args.num_splits
            )
        )

    # ---------------------------------------------------------
    # Shuffle once
    # ---------------------------------------------------------

    rng = np.random.RandomState(args.seed)

    shuffled_keys = keys.copy()
    rng.shuffle(shuffled_keys)

    # ---------------------------------------------------------
    # Create fixed test folds
    # ---------------------------------------------------------

    fold_size = num_videos // args.num_splits

    splits = []

    for split_idx in range(args.num_splits):

        start = split_idx * fold_size
        end = start + fold_size

        # These videos are used ONLY for testing in this fold
        test_keys = shuffled_keys[start:end]

        # Every other video is used for training
        train_keys = [
            key
            for key in shuffled_keys
            if key not in test_keys
        ]

        # -----------------------------------------------------
        # Validation checks
        # -----------------------------------------------------

        assert len(train_keys) + len(test_keys) == num_videos

        assert len(set(train_keys) & set(test_keys)) == 0

        assert len(train_keys) == num_videos - fold_size

        assert len(test_keys) == fold_size

        splits.append({
            'train_keys': sorted(train_keys),
            'test_keys': sorted(test_keys),
        })

    # ---------------------------------------------------------
    # Validate complete 5-fold coverage
    # ---------------------------------------------------------

    all_test_keys = []

    for split in splits:
        all_test_keys.extend(split['test_keys'])

    assert len(all_test_keys) == num_videos

    assert len(set(all_test_keys)) == num_videos, \
        "Error: some videos appear in multiple test folds"

    assert set(all_test_keys) == set(keys), \
        "Error: some videos are missing from test folds"

    # ---------------------------------------------------------
    # Print fold information
    # ---------------------------------------------------------

    print("\n========== FOLD INFORMATION ==========")

    for split_idx, split in enumerate(splits):

        print("\nFold {}".format(split_idx))

        print(
            "Train videos: {}".format(
                len(split['train_keys'])
            )
        )

        print(
            "Test videos : {}".format(
                len(split['test_keys'])
            )
        )

        print(
            "Test keys   : {}".format(
                split['test_keys']
            )
        )

    print("\n======================================")

    # ---------------------------------------------------------
    # Save
    # ---------------------------------------------------------

    saveto = osp.join(
        args.save_dir,
        args.save_name + '.json'
    )

    write_json(splits, saveto)

    print("\nSplits saved to {}".format(saveto))

    dataset.close()


if __name__ == '__main__':
    create()
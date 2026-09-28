# -*- coding: utf-8 -*-

import torch
from torch.utils.data import Dataset, DataLoader
import h5py
import numpy as np
import json
from pathlib import Path


class VideoData(Dataset):

    def __init__(
        self,
        mode,
        split_index,
        cnn_filename,
        semantic_filename,
        splits_filename
    ):
        self.mode = mode
        self.name = 'summe'
        self.split_index = split_index

        cnn_path = Path(cnn_filename).resolve()
        semantic_path = Path(semantic_filename).resolve()
        splits_path = Path(splits_filename).resolve()

        print(f"[DataLoader] Mode: {mode} | Split: {split_index}")
        print(f"[DataLoader] CNN Path: {cnn_path} (Exists: {cnn_path.exists()})")
        print(f"[DataLoader] Semantic Path: {semantic_path} (Exists: {semantic_path.exists()})")
        print(f"[DataLoader] Splits Path: {splits_path} (Exists: {splits_path.exists()})")

        if not cnn_path.exists():
            raise FileNotFoundError(
                f"CNN feature dataset not found at: {cnn_path}\n"
                f"Please verify the file path passed to --dataset_dir."
            )

        if not semantic_path.exists():
            raise FileNotFoundError(
                f"Semantic feature dataset not found at: {semantic_path}\n"
                f"Please verify the file path passed to --semantic_dataset."
            )

        if not splits_path.exists():
            raise FileNotFoundError(
                f"Splits file not found at: {splits_path}\n"
                f"Please verify the file path passed to --splits_file."
            )

        self.video_data = h5py.File(cnn_path, 'r')
        self.semantic_data = h5py.File(semantic_path, 'r')

        with open(splits_path, 'r') as f:
            data = json.load(f)

        self.splits = []

        for split in data:
            self.splits.append({
                'train_keys': split['train_keys'],
                'test_keys': split['test_keys']
            })

    def __len__(self):
        return len(
            self.splits[self.split_index][
                self.mode + '_keys'
            ]
        )

    def __getitem__(self, index):

        video_name = (
            self.splits[self.split_index][
                self.mode + '_keys'
            ][index]
        )

        cnn_features = torch.tensor(
            np.array(
                self.video_data[
                    video_name + '/features'
                ]
            ),
            dtype=torch.float32
        )

        semantic_features = torch.tensor(
            np.array(
                self.semantic_data[
                    video_name + '/features'
                ]
            ),
            dtype=torch.float32
        )

        if cnn_features.ndim != 2:
            raise ValueError(
                'CNN feature tensor must have shape [T, D]. Video: ' + video_name
            )

        if semantic_features.ndim != 2:
            raise ValueError(
                'Semantic feature tensor must have shape [T, D]. Video: ' + video_name
            )

        if cnn_features.size(0) != semantic_features.size(0):
            raise ValueError(
                'CNN and SigLIP2 feature sequence lengths do not match.\n'
                + 'Video: ' + video_name
                + '\nCNN: ' + str(cnn_features.shape)
                + '\nSigLIP2: ' + str(semantic_features.shape)
            )

        if cnn_features.size(-1) != 1024:
            raise ValueError(
                f'Expected CNN dimension 1024 but got {cnn_features.size(-1)} for {video_name}'
            )

        if semantic_features.size(-1) != 768:
            raise ValueError(
                f'Expected SigLIP2 dimension 768 but got {semantic_features.size(-1)} for {video_name}'
            )

        if self.mode == 'test':
            return (
                cnn_features,
                semantic_features,
                video_name
            )

        return (
            cnn_features,
            semantic_features,
            video_name
        )

    def __del__(self):
        try:
            if hasattr(self, 'video_data') and self.video_data:
                self.video_data.close()
        except Exception:
            pass

        try:
            if hasattr(self, 'semantic_data') and self.semantic_data:
                self.semantic_data.close()
        except Exception:
            pass


def get_loader(
    mode,
    split_index,
    cnn_filename,
    semantic_filename,
    splits_filename
):
    dataset = VideoData(
        mode=mode,
        split_index=split_index,
        cnn_filename=cnn_filename,
        semantic_filename=semantic_filename,
        splits_filename=splits_filename
    )

    if mode.lower() == 'train':
        return DataLoader(
            dataset,
            batch_size=1,
            shuffle=True,
            num_workers=0
        )

    return dataset
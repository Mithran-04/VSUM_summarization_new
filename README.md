# Video Summarization Project

This project contains a PyTorch video summarization pipeline for the SumMe and TVSum datasets.

## Project structure

- `train.py` – main training entry point
- `configs.py` – configuration parsing and defaults
- `data_loader.py` – dataset loader and batching
- `dataset_split.py` – generates train/test splits
- `Solver.py` – main solver/training logic
- `Summarizer.py` – summarization inference logic
- `datasets/` – dataset feature files and split JSON files
- `summe/` and `tvsum/` – output folders for logs, checkpoints, and results

## Requirements

Install Python dependencies:

```bash
pip install torch h5py numpy
```

## Prepare dataset splits

Generate the 5-fold split JSON for SumMe:

```bash
python dataset_split.py --dataset datasets/summe/eccv16_dataset_summe_google_pool5.h5 --save-dir datasets/summe/splits --save-name summe_splits --num-splits 5
```

Generate the 5-fold split JSON for TVSum:

```bash
python dataset_split.py --dataset datasets/tvsum/eccv16_dataset_tvsum_google_pool5.h5 --save-dir datasets/tvsum/splits --save-name tvsum_splits --num-splits 5
```

## Train the model

Train on SumMe:

```bash
python train.py --dataset_dir datasets/summe/eccv16_dataset_summe_google_pool5.h5 --semantic_dataset datasets/summe/eccv16_dataset_summe_siglip2.h5 --splits_file datasets/summe/splits/summe_splits.json --split_index 0 --video_type summe
```

Train on TVSum:

```bash
python train.py --dataset_dir "./datasets/tvsum/eccv16_dataset_tvsum_google_pool5_with_names.h5" --semantic_dataset "./datasets/tvsum/eccv16_dataset_tvsum_siglip2.h5" --splits_file "./datasets/tvsum/splits/tvsum_splits.json" --video_type tvsum --split_index 0 --n_epochs 80 --lr 0.0001 --lr_scorer 0.0001
```

## Notes

- The repo expects HDF5 datasets with feature arrays shaped for the CNN and semantic embeddings.
- Training outputs are saved under the dataset-specific output folders such as `summe/models/...` or `tvsum/results/...`.
- If a required file path is missing, the script raises a `FileNotFoundError` with the exact path it was looking for.

## Common issue

If the dataset files are not available in the expected path, double-check the file names and the folder structure before running the training command.

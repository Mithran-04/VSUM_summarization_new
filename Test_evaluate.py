import argparse
import json
from pathlib import Path

import h5py
import numpy as np
import torch

from configs import get_config
from Summarizer import Summarizer
from vsum_tools import generate_summary, evaluate_summary


def read_scalar(value):
    """Safely unpack HDF5 scalar dataset to standard Python int/float."""
    val = np.asarray(value)
    if val.size == 1:
        return val.item()
    return val


def find_video_group(h5_file, video_key):
    """Find a video group by its H5 key."""
    if video_key not in h5_file:
        raise KeyError(f"Video group {video_key!r} not found in HDF5")
    return h5_file[video_key]


def extract_model_scores(output):
    """Get final frame scores from the current Summarizer output."""
    if isinstance(output, dict):
        if "final_scores" in output:
            scores = output["final_scores"]
        elif "scores" in output:
            scores = output["scores"]
        else:
            raise KeyError("Summarizer output must contain 'final_scores' or 'scores'.")
    elif isinstance(output, (tuple, list)):
        scores = output[0]
    else:
        scores = output

    return scores


def load_model(config, checkpoint_path, device):
    model = Summarizer(
        cnn_size=config.input_size,
        semantic_size=config.semantic_size,
        hidden_size=config.hidden_size,
        d_state=config.d_state,
        dropout=getattr(config, "dropout", 0.1)
    ).to(device)

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False
    )

    if isinstance(checkpoint, dict):
        if "summarizer" in checkpoint:
            state_dict = checkpoint["summarizer"]
        elif "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
        elif "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        else:
            state_dict = checkpoint
    else:
        state_dict = checkpoint

    state_dict = {
        key.removeprefix("module."): value
        for key, value in state_dict.items()
    }

    model.load_state_dict(state_dict, strict=True)
    model.eval()

    return model


def evaluate_dataset(
    dataset_name,
    cnn_h5_path,
    semantic_h5_path,
    checkpoint_path,
    output_h5_path,
    config,
    splits_file=None,
    split_index=0
):
    assert dataset_name in ("summe", "tvsum")

    eval_metric = "max" if dataset_name == "summe" else "avg"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"\nEvaluating {dataset_name.upper()} on {device}")

    model = load_model(config, checkpoint_path, device)
    fms, precisions, recalls = [], [], []

    with h5py.File(cnn_h5_path, "r") as cnn_h5, \
         h5py.File(semantic_h5_path, "r") as semantic_h5, \
         h5py.File(output_h5_path, "w") as result_h5:

        # Check if splits_file exists and parse test keys
        if splits_file and Path(splits_file).is_file():
            with open(splits_file, "r") as f:
                splits = json.load(f)

            if split_index < 0 or split_index >= len(splits):
                raise ValueError(
                    f"split_index {split_index} out of range for {len(splits)} splits in {splits_file}"
                )

            raw_test_keys = splits[split_index]["test_keys"]
            all_h5_keys = sorted(list(cnn_h5.keys()))

            # --- DIAGNOSTIC PRINTS ---
            print("\n=== DIAGNOSTIC KEY CHECK ===")
            print(f"HDF5 total keys: {len(all_h5_keys)}")
            print(f"HDF5 sample keys: {all_h5_keys[:3]}")
            print(f"Splits test_keys count: {len(raw_test_keys)}")
            print(f"Splits sample test_keys: {raw_test_keys[:3]}")
            print("============================\n")

            # Try direct string matching first
            test_keys_set = set(str(k) for k in raw_test_keys)
            video_keys = [k for k in all_h5_keys if k in test_keys_set]

            # Fallback: Handle "video_1", "video_2", etc. as 1-based indices into sorted H5 keys
            if len(video_keys) == 0 and all(isinstance(k, str) and k.startswith("video_") for k in raw_test_keys):
                print("[INFO] Direct string matching failed. Attempting to map 'video_X' keys as index offsets...")
                try:
                    indices = [int(k.replace("video_", "")) - 1 for k in raw_test_keys]
                    video_keys = [all_h5_keys[idx] for idx in indices if 0 <= idx < len(all_h5_keys)]
                    print(f"[SUCCESS] Mapped {len(video_keys)} videos using index offsets!")
                except (ValueError, IndexError) as e:
                    print(f"[ERROR] Index mapping failed: {e}")

            if len(video_keys) == 0:
                print(f"[WARNING] None of the test_keys in split {split_index} matched keys in {cnn_h5_path}!")
                raise RuntimeError("Key matching failed between splits file and HDF5 file.")

            print(f"Loaded split {split_index}: Evaluating ONLY {len(video_keys)} test videos out of {len(all_h5_keys)}")
        else:
            video_keys = sorted(cnn_h5.keys())
            print(f"[INFO] No valid splits file found at '{splits_file}'. Evaluating on ALL {len(video_keys)} videos.")

        for video_key in video_keys:
            cnn_group = find_video_group(cnn_h5, video_key)
            sem_group = find_video_group(semantic_h5, video_key)

            cnn_features = cnn_group["features"][...].astype(np.float32)
            semantic_features = sem_group["features"][...].astype(np.float32)

            cnn_tensor = torch.from_numpy(cnn_features).unsqueeze(0).to(device)
            semantic_tensor = torch.from_numpy(semantic_features).unsqueeze(0).to(device)

            with torch.no_grad():
                output = model(cnn_tensor, semantic_tensor)
                scores = extract_model_scores(output)
                scores = scores.detach().cpu().numpy().reshape(-1)

            cps = cnn_group["change_points"][...]
            n_frames = read_scalar(cnn_group["n_frames"][()])
            positions = cnn_group["picks"][...].astype(np.int32)
            user_summary = cnn_group["user_summary"][...]

            # FIX 1: Format nfps as a list of native Python ints for knapsack check_inputs
            nfps = [int(x) for x in cnn_group["n_frame_per_seg"][...].reshape(-1)]

            # FIX 2: Align scores strictly to len(positions) without modifying positions
            target_pos_len = len(positions)
            if len(scores) > target_pos_len:
                scores = scores[:target_pos_len]
                print("Yesssssssss error")
            elif len(scores) < target_pos_len:
                scores = np.interp(
                    np.linspace(0, len(scores) - 1, target_pos_len),
                    np.arange(len(scores)),
                    scores
                )
                print("Yessssssssssssssssss error")

            machine_summary = generate_summary(
                ypred=scores,
                cps=cps,
                n_frames=n_frames,
                nfps=nfps,
                positions=positions,
                proportion=0.15
            )

            fm, precision, recall = evaluate_summary(
                machine_summary,
                user_summary,
                eval_metric=eval_metric
            )

            fms.append(fm)
            precisions.append(precision)
            recalls.append(recall)

            print(f"{video_key}: F={fm:.4f}, P={precision:.4f}, R={recall:.4f}")

            result_group = result_h5.create_group(video_key)
            result_group.create_dataset("score", data=scores)
            result_group.create_dataset("machine_summary", data=machine_summary)
            if "gtscore" in cnn_group:
                result_group.create_dataset("gtscore", data=cnn_group["gtscore"][...])
            result_group.create_dataset("fm", data=fm)
            result_group.create_dataset("precision", data=precision)
            result_group.create_dataset("recall", data=recall)

        mean_fm = float(np.mean(fms))
        mean_precision = float(np.mean(precisions))
        mean_recall = float(np.mean(recalls))

        result_h5.attrs["dataset"] = dataset_name
        result_h5.attrs["eval_metric"] = eval_metric
        result_h5.attrs["mean_fmeasure"] = mean_fm
        result_h5.attrs["mean_precision"] = mean_precision
        result_h5.attrs["mean_recall"] = mean_recall

    print("\n========== RESULTS ==========")
    print(f"Dataset:   {dataset_name.upper()}")
    print(f"Split:     {split_index if splits_file else 'All'}")
    print(f"Metric:    {eval_metric}")
    print(f"F-measure: {mean_fm:.4f} ({mean_fm * 100:.2f}%)")
    print(f"Precision: {mean_precision:.4f}")
    print(f"Recall:    {mean_recall:.4f}")
    print(f"Saved to:  {output_h5_path}")

    return mean_fm, mean_precision, mean_recall


if __name__ == "__main__":
    import sys

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["summe", "tvsum"], required=True)
    parser.add_argument("--cnn-h5", required=True)
    parser.add_argument("--semantic-h5", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--splits-file", type=str, default=None)
    parser.add_argument("--split-index", type=int, default=0)

    args, _ = parser.parse_known_args()

    saved_argv = sys.argv
    sys.argv = [
        saved_argv[0],
        "--dataset_dir", "dummy",
        "--semantic_dataset", "dummy",
        "--splits_file", "dummy"
    ]
    config = get_config(mode="test")
    sys.argv = saved_argv

    evaluate_dataset(
        dataset_name=args.dataset,
        cnn_h5_path=args.cnn_h5,
        semantic_h5_path=args.semantic_h5,
        checkpoint_path=args.checkpoint,
        output_h5_path=args.output,
        config=config,
        splits_file=args.splits_file,
        split_index=args.split_index
    )
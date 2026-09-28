# -*- coding: utf-8 -*-

import h5py
import cv2
import torch
import numpy as np

from pathlib import Path
from PIL import Image
from tqdm import tqdm
from transformers import AutoProcessor, AutoModel


# ============================================================
# CONFIG
# ============================================================

GOOGLENET_H5 = Path(
    "eccv16_dataset_summe_google_pool5.h5"
)

VIDEO_DIR = Path(
    "data/DataSet/SumMeFrames/videos"
)

OUTPUT_H5 = Path(
    "eccv16_dataset_summe_siglip2.h5"
)

# SigLIP 2
MODEL_NAME = "google/siglip2-base-patch16-224"

BATCH_SIZE = 16

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ============================================================
# VIDEO NAME MATCHING
# ============================================================

def normalize_name(name):
    """
    Convert names such as:

        Air_Force_One
        Air Force One
        Air-Force-One.mp4

    into the same searchable name.
    """

    name = Path(name).stem

    return "".join(
        c.lower()
        for c in name
        if c.isalnum()
    )


def find_video(video_name, video_files):

    target = normalize_name(video_name)

    for video_path in video_files:

        if normalize_name(video_path.name) == target:
            return video_path

    raise FileNotFoundError(
        f"Video not found: {video_name}"
    )


# ============================================================
# LOAD SELECTED FRAMES
# ============================================================

def get_frames(video_path, picks):
    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    picks = set(int(x) for x in picks)
    frames = []

    frame_number = 0

    while True:
        success, frame = cap.read()

        if not success:
            break

        if frame_number in picks:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(Image.fromarray(frame))

            if len(frames) == len(picks):
                break

        frame_number += 1

    cap.release()

    if len(frames) != len(picks):
        raise RuntimeError(
            f"Could not read all required frames from {video_path}. "
            f"Expected {len(picks)}, got {len(frames)}."
        )

    return frames


# ============================================================
# SIGLIP 2 FEATURE EXTRACTION
# ============================================================

def get_siglip2_features(
    frames,
    processor,
    model
):

    features = []

    for i in range(
        0,
        len(frames),
        BATCH_SIZE
    ):

        batch = frames[
            i:i + BATCH_SIZE
        ]

        inputs = processor(
            images=batch,
            return_tensors="pt"
        )

        inputs = {
            key: value.to(DEVICE)
            for key, value in inputs.items()
        }

        with torch.no_grad():

            output = model.get_image_features(
                **inputs
            )

        # SigLIP 2 may return a model output object
        # instead of a tensor depending on the
        # Transformers version.
        if hasattr(output, "pooler_output"):
            output = output.pooler_output

        elif hasattr(output, "last_hidden_state"):
            output = output.last_hidden_state[:, 0]

        # L2 normalization
        output = torch.nn.functional.normalize(
            output,
            p=2,
            dim=-1
        )

        features.append(
            output.cpu().numpy()
        )

    return np.concatenate(
        features,
        axis=0
    ).astype(np.float32)


# ============================================================
# MAIN
# ============================================================

def main():

    print("Device:", DEVICE)
    print("Loading SigLIP 2...")

    processor = AutoProcessor.from_pretrained(
        MODEL_NAME
    )

    model = AutoModel.from_pretrained(
        MODEL_NAME
    ).to(DEVICE)

    model.eval()

    print("SigLIP 2 loaded.")

    # --------------------------------------------------------
    # Find all videos
    # --------------------------------------------------------

    video_files = [
        p
        for p in VIDEO_DIR.rglob("*")
        if p.is_file()
        and p.suffix.lower()
        in [
            ".mp4",
            ".avi",
            ".mov",
            ".mkv",
            ".webm"
        ]
    ]

    print(
        "Videos found:",
        len(video_files)
    )

    # --------------------------------------------------------
    # Open H5 files
    # --------------------------------------------------------

    with h5py.File(
        GOOGLENET_H5,
        "r"
    ) as input_h5, h5py.File(
        OUTPUT_H5,
        "w"
    ) as output_h5:

        video_names = list(input_h5.keys())

        # ----------------------------------------------------
        # Process each video
        # ----------------------------------------------------

        for h5_name in tqdm(
            video_names,
            desc="Videos"
        ):

            group = input_h5[h5_name]

            # ------------------------------------------------
            # Get original video name
            # ------------------------------------------------

            video_name = group[
                "video_name"
            ][()]

            if isinstance(
                video_name,
                bytes
            ):
                video_name = video_name.decode()
            # if video_name == "playing_ball":
            #     print(f"Skipping {video_name}")
            #     continue

            # ------------------------------------------------
            # Find original video
            # ------------------------------------------------
            print("Video_name: ",video_name)
            video_path = find_video(
                video_name,
                video_files
            )
            print("Video_path: ",video_path)

            # ------------------------------------------------
            # Get exact frames used for GoogLeNet
            # ------------------------------------------------

            picks = group[
                "picks"
            ][:]

            # ------------------------------------------------
            # Extract those frames
            # ------------------------------------------------

            frames = get_frames(
                video_path,
                picks
            )

            # ------------------------------------------------
            # Extract SigLIP 2 features
            # ------------------------------------------------

            siglip_features = get_siglip2_features(
                frames,
                processor,
                model
            )

            print(
                f"\n{video_name}: "
                f"{siglip_features.shape}"
            )

            # ------------------------------------------------
            # Create output group
            # ------------------------------------------------

            output_group = output_h5.create_group(
                h5_name
            )

            # ------------------------------------------------
            # Copy existing metadata
            # ------------------------------------------------

            for key in group.keys():

                if key == "features":
                    continue

                group.copy(
                    key,
                    output_group
                )

            # ------------------------------------------------
            # Save SigLIP 2 features
            # ------------------------------------------------

            output_group.create_dataset(
                "features",
                data=siglip_features,
                compression="gzip"
            )

            output_h5.flush()

    print("\nFinished.")
    print(
        "Saved:",
        OUTPUT_H5.resolve()
    )


if __name__ == "__main__":
    main()
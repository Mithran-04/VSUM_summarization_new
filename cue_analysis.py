# cue_analysis.py
import numpy as np, h5py
from scipy.stats import spearmanr

cnn_path = "./datasets/summe/eccv16_dataset_summe_google_pool5.h5"
sem_path = "./datasets/summe/eccv16_dataset_summe_siglip2.h5"

def cues(f):                                   # f: [T, D] features
    f = f / (np.linalg.norm(f, axis=1, keepdims=True) + 1e-8)
    mean_dist = 1 - f @ (f.mean(0) / np.linalg.norm(f.mean(0)))     # distinctiveness
    nb = np.zeros(len(f)); nb[1:] = 1 - (f[1:] * f[:-1]).sum(1)     # novelty vs previous frame
    return {"dist_to_mean": mean_dist, "neighbor_change": nb}

res = {}
with h5py.File(cnn_path, "r") as cnn, h5py.File(sem_path, "r") as sem:
    for i, k in enumerate(cnn.keys()):
        gt = cnn[k]["gtscore"][...]
        pos = cnn[k]["picks"][...]
        if i == 0:
            print("len(gtscore) =", len(gt), "| len(picks) =", len(pos))   # check once
        if len(gt) != len(pos):
            gt = gt[pos]                       # match resolution
        for name, feats in [("cnn", cnn[k]["features"][...]),
                            ("siglip", sem[k]["features"][...])]:
            for cname, c in cues(feats).items():
                res.setdefault(f"{name}/{cname}", []).append(spearmanr(c, gt).correlation)

for k, v in res.items():
    print(f"{k:28s} mean rho = {np.nanmean(v):+.3f}")
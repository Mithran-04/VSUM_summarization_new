import sys
import h5py
import numpy as np
from scipy.stats import spearmanr, kendalltau

path = sys.argv[1] if len(sys.argv) > 1 else "./summe/test/result_test.h5"
rng = np.random.default_rng(0)

rhos, taus, null_rhos = [], [], []
with h5py.File(path, "r") as f:
    for key in sorted(f.keys()):
        if "gtscore" not in f[key]:
            continue
        score = f[key]["score"][...].reshape(-1).astype(np.float64)
        gt = f[key]["gtscore"][...].reshape(-1).astype(np.float64)

        # gtscore may be per-frame while score is per-sampled-frame: resample gt to score length
        if len(gt) != len(score):
            gt = np.interp(np.linspace(0, len(gt) - 1, len(score)),
                           np.arange(len(gt)), gt)

        if np.std(gt) == 0 or np.std(score) == 0:
            print(f"{key}: constant scores, skipped")
            continue

        rho = spearmanr(score, gt).correlation
        tau = kendalltau(score, gt).correlation
        # chance level for this video: shuffled scores
        null = [spearmanr(rng.permutation(score), gt).correlation for _ in range(200)]

        rhos.append(rho); taus.append(tau); null_rhos.append(np.std(null))
        print(f"{key}: spearman={rho:+.3f}  kendall={tau:+.3f}  (chance sd ~{np.std(null):.3f})")

print("\n===== SUMMARY =====")
print(f"videos:          {len(rhos)}")
print(f"mean Spearman:   {np.mean(rhos):+.3f}")
print(f"mean Kendall:    {np.mean(taus):+.3f}")
print(f"videos with rho>0: {sum(r > 0 for r in rhos)}/{len(rhos)}")
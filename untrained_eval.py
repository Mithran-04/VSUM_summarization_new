import sys, csv
import numpy as np
import torch
from pathlib import Path
from configs import get_config
from Summarizer import Summarizer
from Test_evaluate import evaluate_dataset

CNN = './datasets/summe/eccv16_dataset_summe_google_pool5.h5'
SEM = './datasets/summe/eccv16_dataset_summe_siglip2.h5'
N_SEEDS = 1

# build the config the same way Test_evaluate.py does
saved = sys.argv
sys.argv = [saved[0], '--dataset_dir', 'dummy', '--semantic_dataset', 'dummy', '--splits_file', 'dummy']
config = get_config(mode='test')
sys.argv = saved

out_dir = Path('./summe/test'); out_dir.mkdir(parents=True, exist_ok=True)
ckpt = out_dir / 'untrained.pth'
csv_path = out_dir / 'untrained_f1.csv'
csv_path.unlink(missing_ok=True)

f1s = []
for seed in range(N_SEEDS):
    torch.manual_seed(seed); np.random.seed(seed)
    model = Summarizer(cnn_size=config.input_size, semantic_size=config.semantic_size,
                       hidden_size=config.hidden_size, d_state=config.d_state,
                       dropout=config.dropout, summary_rate=config.summary_rate)
    torch.save({'model_state_dict': model.state_dict()}, ckpt)       # random weights, no training

    fm, p, r = evaluate_dataset(
        dataset_name='summe', cnn_h5_path=CNN, semantic_h5_path=SEM,
        checkpoint_path=str(ckpt), output_h5_path=str(out_dir / 'untrained_result.h5'),
        config=config, splits_file=None)                             # None = all 25 videos

    f1s.append(fm * 100)
    with open(csv_path, 'a', newline='') as f:
        w = csv.writer(f)
        if seed == 0:
            w.writerow(['seed', 'f1', 'precision', 'recall'])
        w.writerow([seed, round(fm * 100, 2), round(p * 100, 2), round(r * 100, 2)])

print(f'\nUntrained model over {N_SEEDS} seeds: F1 = {np.mean(f1s):.2f} ± {np.std(f1s):.2f}')
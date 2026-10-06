# -*- coding: utf-8 -*-

import csv
import json
import shutil
import subprocess
import sys
from pathlib import Path

import h5py
import torch
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm, trange

from Summarizer import Summarizer
from Attention import FeatureDiscriminator

# ---------------------------------------------------------------------------
# Run settings
# ---------------------------------------------------------------------------
EVAL_DATASET = 'tvsum'          # 'summe' or 'tvsum'
RUN_TAG = 'fssa_mamba_adv_v1'   # appears in checkpoint / csv names so old runs are not overwritten

EVAL_PATHS = {
    'summe': {
        'out_dir': './summe/test',
        'cnn_h5': './datasets/summe/eccv16_dataset_summe_google_pool5.h5',
        'semantic_h5': './datasets/summe/eccv16_dataset_summe_siglip2.h5',
        'splits': './datasets/summe/splits/summe_splits.json',
    },
    'tvsum': {
        'out_dir': './tvsum/test',
        'cnn_h5': './datasets/tvsum/eccv16_dataset_tvsum_google_pool5_with_names.h5',
        'semantic_h5': './datasets/tvsum/eccv16_dataset_tvsum_siglip2.h5',
        'splits': './datasets/tvsum/splits/tvsum_splits.json',
    },
}


class Solver:
    def __init__(self, config, train_loader, test_loader):
        self.config = config
        self.train_loader = train_loader
        self.test_loader = test_loader

        self.previous_selected = None
        self.best_f1 = -1.0
        self.best_epoch = None

        self.config.save_dir.mkdir(parents=True, exist_ok=True)
        self.metrics_csv = self.config.save_dir / (
            f'{self.config.video_type}_{RUN_TAG}_training_metrics_split_{self.config.split_index}.csv')
        if not self.metrics_csv.exists():
            with open(self.metrics_csv, 'w', newline='') as f:
                csv.writer(f).writerow([
                    'split', 'epoch', 'total_loss', 'reconstruction_loss', 'sparsity_loss',
                    'adv_loss', 'disc_loss', 'd_real', 'd_fake', 'adv_weight',
                    'alpha', 'score_std', 'topk_jaccard'
                ])

    # ------------------------------------------------------------------
    def build(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print('Using device:', self.device)

        self.summarizer = Summarizer(
            cnn_size=self.config.input_size, semantic_size=self.config.semantic_size,
            hidden_size=self.config.hidden_size, d_state=self.config.d_state,
            dropout=self.config.dropout, summary_rate=self.config.summary_rate
        ).to(self.device)

        scorer_params = list(self.summarizer.scorer.parameters())
        scorer_ids = {id(p) for p in scorer_params}
        other_params = [p for p in self.summarizer.parameters() if id(p) not in scorer_ids]
        self.optimizer = optim.AdamW([
            {'params': other_params, 'lr': self.config.lr},
            {'params': scorer_params, 'lr': self.config.lr_scorer},
        ], weight_decay=self.config.weight_decay)

        # Discriminator (training only; not part of the Summarizer / test-time checkpoint weights)
        self.discriminator = FeatureDiscriminator(
            in_size=self.config.input_size + self.config.semantic_size
        ).to(self.device)
        self.optimizer_d = optim.AdamW(
            self.discriminator.parameters(),
            lr=getattr(self.config, 'lr_disc', 1e-4),
            weight_decay=self.config.weight_decay)

        total_parameters = sum(p.numel() for p in self.summarizer.parameters() if p.requires_grad)
        print('Trainable parameters (summarizer):', total_parameters)
        print('Trainable parameters (discriminator):',
              sum(p.numel() for p in self.discriminator.parameters() if p.requires_grad))

    # ------------------------------------------------------------------
    # Generator-side losses: L = lambda_recon * L_rec + lambda_sparse * L_sparsity
    #                          + w_adv(epoch) * L_adv
    # ------------------------------------------------------------------
    def reconstruction_loss(self, target, recon):
        c = self.config.input_size
        mse_c = ((recon[..., :c] - target[..., :c]) ** 2).sum(-1).mean()
        mse_s = ((recon[..., c:] - target[..., c:]) ** 2).sum(-1).mean()
        return mse_c + mse_s

    def sparsity_loss(self, scores):
        return torch.abs(scores.mean(dim=1) - self.config.summary_rate).mean()

    def total_loss(self, outputs):
        rec = self.reconstruction_loss(outputs['target'], outputs['reconstructed_features'])
        spr = self.sparsity_loss(outputs['scores'])
        lambda_recon = getattr(self.config, 'lambda_recon', 1.0)
        lambda_sparse = getattr(self.config, 'lambda_sparse', 1.0)
        total = lambda_recon * rec + lambda_sparse * spr
        return total, {'reconstruction': rec, 'sparsity': spr}

    # ------------------------------------------------------------------
    # Adversarial part
    # ------------------------------------------------------------------
    def _normalize_pair(self, x):
        """L2-normalize CNN and SigLIP2 parts separately (same space as the real target)."""
        c = self.config.input_size
        return torch.cat([F.normalize(x[..., :c], dim=-1),
                          F.normalize(x[..., c:], dim=-1)], dim=-1)

    def adv_weight(self, epoch_i):
        """0 before adv_start_epoch, then ramps linearly to lambda_adv over adv_ramp_epochs."""
        lambda_adv = getattr(self.config, 'lambda_adv', 0.1)
        start = getattr(self.config, 'adv_start_epoch', 10)
        ramp = max(1, getattr(self.config, 'adv_ramp_epochs', 5))
        if epoch_i < start:
            return 0.0
        return lambda_adv * min(1.0, (epoch_i - start + 1) / ramp)

    def discriminator_step(self, real, fake):
        """Hinge loss: push D(real) above +1 and D(fake) below -1. Updates D only."""
        self.optimizer_d.zero_grad()
        d_real = self.discriminator(real)
        d_fake = self.discriminator(fake.detach())
        d_loss = F.relu(1.0 - d_real).mean() + F.relu(1.0 + d_fake).mean()
        d_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.discriminator.parameters(), self.config.clip)
        self.optimizer_d.step()
        return d_loss.detach().item(), d_real.detach().mean().item(), d_fake.detach().mean().item()

    def generator_adv_loss(self, fake):
        """Generator side of the hinge loss: -E[D(fake)]. D weights are frozen for this forward."""
        for p in self.discriminator.parameters():
            p.requires_grad_(False)
        adv = -self.discriminator(fake).mean()
        for p in self.discriminator.parameters():
            p.requires_grad_(True)
        return adv

    # ------------------------------------------------------------------
    @staticmethod
    def selection_jaccard(current_selected, previous_selected):
        values = []
        for name in set(current_selected) & set(previous_selected):
            cur, prev = set(current_selected[name]), set(previous_selected[name])
            union = cur | prev
            values.append(1.0 if not union else len(cur & prev) / len(union))
        return sum(values) / len(values) if values else float('nan')

    # ------------------------------------------------------------------
    def train(self):
        for epoch_i in trange(self.config.n_epochs, desc='Epoch'):
            self.summarizer.train()
            self.discriminator.train()
            w_adv = self.adv_weight(epoch_i)

            sums = {'total': 0.0, 'reconstruction': 0.0, 'sparsity': 0.0, 'adv': 0.0,
                    'disc': 0.0, 'd_real': 0.0, 'd_fake': 0.0, 'alpha': 0.0, 'score_std': 0.0}
            epoch_selected = {}

            for batch in tqdm(self.train_loader, desc='Batch', leave=False):
                cnn_features = batch[0].to(self.device)
                semantic_features = batch[1].to(self.device)
                video_name = batch[2][0]

                outputs = self.summarizer(cnn_features, semantic_features)
                gen_loss, losses = self.total_loss(outputs)

                adv_val = disc_val = d_real_val = d_fake_val = 0.0
                if w_adv > 0:
                    real = outputs['target'].detach()
                    fake = self._normalize_pair(outputs['reconstructed_features'])

                    # 1) discriminator update (real vs detached fake)
                    disc_val, d_real_val, d_fake_val = self.discriminator_step(real, fake)

                    # 2) generator adversarial term (gradient flows to scorer + decoder)
                    adv_loss = self.generator_adv_loss(fake)
                    adv_val = adv_loss.detach().item()
                    gen_loss = gen_loss + w_adv * adv_loss

                # 3) selector + generator update
                self.optimizer.zero_grad()
                gen_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.summarizer.parameters(), self.config.clip)
                self.optimizer.step()

                scores = outputs['scores'].detach()
                score_std = scores.std(dim=1).mean().item()
                k = max(1, int(round(scores.size(1) * self.config.summary_rate)))
                epoch_selected[video_name] = scores.squeeze(-1)[0].topk(k).indices.cpu().tolist()

                sums['total'] += gen_loss.detach().item()
                sums['reconstruction'] += losses['reconstruction'].detach().item()
                sums['sparsity'] += losses['sparsity'].detach().item()
                sums['adv'] += adv_val
                sums['disc'] += disc_val
                sums['d_real'] += d_real_val
                sums['d_fake'] += d_fake_val
                sums['alpha'] += outputs['alpha'].detach().item()
                sums['score_std'] += score_std

                if self.config.verbose:
                    print(f"\n[{video_name}] epoch {epoch_i} | gen={gen_loss.item():.4f} "
                          f"rec={losses['reconstruction'].item():.4f} "
                          f"spr={losses['sparsity'].item():.4f} adv={adv_val:.4f} "
                          f"D={disc_val:.4f} (real={d_real_val:.2f}, fake={d_fake_val:.2f}) | "
                          f"S mean={scores.mean().item():.4f} std={score_std:.4f} "
                          f"alpha={outputs['alpha'].item():.3f}")

            n = max(1, len(self.train_loader))
            avg = {key: val / n for key, val in sums.items()}

            if self.previous_selected is None:
                topk_jaccard = float('nan')
            else:
                topk_jaccard = self.selection_jaccard(epoch_selected, self.previous_selected)
            self.previous_selected = {name: list(idx) for name, idx in epoch_selected.items()}

            with open(self.metrics_csv, 'a', newline='') as f:
                csv.writer(f).writerow([
                    self.config.split_index, epoch_i, avg['total'], avg['reconstruction'],
                    avg['sparsity'], avg['adv'], avg['disc'], avg['d_real'], avg['d_fake'],
                    w_adv, avg['alpha'], avg['score_std'], topk_jaccard
                ])

            print(f'\n================ Epoch {epoch_i} Completed (adv weight {w_adv:.3f}) ================')
            for key, val in avg.items():
                print(f'{key}: {val:.4f}')
            print(f'topk_jaccard: {topk_jaccard:.4f}')
            print('========================================================')

            ckpt = self.save_checkpoint(epoch_i)
            self.run_test_evaluate(epoch_i, ckpt)

    # ------------------------------------------------------------------
    def evaluate(self, epoch_i):
        self.summarizer.eval()
        results = {}

        with torch.no_grad():
            for batch in tqdm(self.test_loader, desc='Evaluate', leave=False):
                cnn_features = batch[0].to(self.device)
                semantic_features = batch[1].to(self.device)
                video_name = batch[2][0]

                outputs = self.summarizer(cnn_features, semantic_features)
                results[video_name] = outputs['scores'].squeeze(-1)[0].cpu().numpy().tolist()

        self.config.score_dir.mkdir(parents=True, exist_ok=True)
        output_file = self.config.score_dir / f'{self.config.video_type}_{RUN_TAG}_epoch_{epoch_i}.json'
        with open(output_file, 'w') as f:
            json.dump(results, f)
        print('Saved scores to:', output_file)

    # ------------------------------------------------------------------
    def run_test_evaluate(self, epoch_i, ckpt):
        paths = EVAL_PATHS[EVAL_DATASET]
        out_dir = Path(paths['out_dir'])
        out_dir.mkdir(parents=True, exist_ok=True)
        out_h5 = out_dir / f'result_test_{RUN_TAG}_split_{self.config.split_index}.h5'

        cmd = [sys.executable, 'Test_evaluate.py',
               '--dataset', EVAL_DATASET,
               '--cnn-h5', paths['cnn_h5'],
               '--semantic-h5', paths['semantic_h5'],
               '--checkpoint', str(ckpt),
               '--output', str(out_h5),
               '--splits-file', paths['splits'],
               '--split-index', str(self.config.split_index)]

        proc = subprocess.run(cmd, capture_output=True, text=True)

        f1 = precision = recall = 'NA'
        try:
            with h5py.File(out_h5, 'r') as f:
                f1 = round(float(f.attrs['mean_fmeasure']) * 100, 2)
                precision = round(float(f.attrs['mean_precision']) * 100, 2)
                recall = round(float(f.attrs['mean_recall']) * 100, 2)
        except Exception as e:
            print(f'[eval] could not read result (returncode={proc.returncode}): {e}')
            print('\n'.join((proc.stdout + proc.stderr).strip().splitlines()[-15:]))

        csv_path = out_dir / f'epoch_f1_{RUN_TAG}_split_{self.config.split_index}.csv'
        new_file = not csv_path.exists()
        with open(csv_path, 'a', newline='') as f:
            w = csv.writer(f)
            if new_file:
                w.writerow(['split', 'epoch', 'f1', 'precision', 'recall'])
            w.writerow([self.config.split_index, epoch_i, f1, precision, recall])
        print(f'[eval] epoch {epoch_i}: F1 = {f1}')

        # NOTE: picking the best epoch on test F1 is optimistic; for reported numbers
        # prefer a fixed epoch count or a held-out validation split.
        if isinstance(f1, (int, float)) and f1 > self.best_f1:
            self.best_f1, self.best_epoch = f1, epoch_i
            best_path = Path(ckpt).with_name(Path(ckpt).stem + '_best_f1.pth')
            shutil.copyfile(ckpt, best_path)
            print(f'[eval] new best F1 = {f1} at epoch {epoch_i} -> {best_path}')

    # ------------------------------------------------------------------
    def save_checkpoint(self, epoch_i):
        self.config.save_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_path = self.config.save_dir / (
            f'{self.config.video_type}_{RUN_TAG}_split_{self.config.split_index}.pth')

        torch.save({
            'epoch': epoch_i,
            'model_state_dict': self.summarizer.state_dict(),          # used by Test_evaluate.py
            'optimizer_state_dict': self.optimizer.state_dict(),
            'disc_state_dict': self.discriminator.state_dict(),
            'optimizer_d_state_dict': self.optimizer_d.state_dict()
        }, checkpoint_path)
        print('Saved checkpoint:', checkpoint_path)
        return checkpoint_path
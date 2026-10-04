# -*- coding: utf-8 -*-

import json
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm, trange
from Summarizer import Summarizer
import torch.nn.functional as F
import csv
import re
import subprocess
import sys
from pathlib import Path
import h5py
import shutil

class Solver:
    def __init__(self, config, train_loader, test_loader):
        self.config = config
        self.train_loader = train_loader
        self.test_loader = test_loader

        # Previous epoch's Top-K selections, indexed by video name
        self.previous_selected = None
        self.best_f1 = -1.0
        self.best_epoch = None

        # Training metrics CSV
        self.metrics_csv = (self.config.save_dir /f'{self.config.video_type}_training_metrics_split_{self.config.split_index}_v3_1.csv')
        self.config.save_dir.mkdir(parents=True, exist_ok=True)
        if not self.metrics_csv.exists():
            with open(self.metrics_csv, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    'split',
                    'epoch',
                    'total_loss',
                    'reconstruction_loss',
                    'diversity_loss',
                    'representativeness_loss',
                    'smoothness_loss',
                    'topk_jaccard'
                ])



    def build(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print('Using device:', self.device)

        self.summarizer = Summarizer(
            cnn_size=self.config.input_size, semantic_size=self.config.semantic_size,
            hidden_size=self.config.hidden_size, d_state=self.config.d_state,
            dropout=self.config.dropout, summary_rate=self.config.summary_rate
        ).to(self.device)

        scorer_params = list(self.summarizer.interceptor_scorer.parameters())
        scorer_ids = {id(p) for p in scorer_params}
        other_params = [p for p in self.summarizer.parameters() if id(p) not in scorer_ids]
        self.optimizer = optim.AdamW([
            {'params': other_params,  'lr': self.config.lr},
            {'params': scorer_params, 'lr': self.config.lr_scorer},
        ], weight_decay=self.config.weight_decay)

        total_parameters = sum(
            p.numel() for p in self.summarizer.parameters() if p.requires_grad
        )
        print('Trainable parameters:', total_parameters)

     # Losses
    def reconstruction_loss(self, target, recon):
        c = self.config.input_size
        l_cnn = 1 - F.cosine_similarity(recon[..., :c], target[..., :c], dim=-1).mean()
        l_sem = 1 - F.cosine_similarity(recon[..., c:], target[..., c:], dim=-1).mean()
        return l_cnn + l_sem

    def diversity_loss(self, feats, idx):
        # mean pairwise cosine similarity among selected frames (repulsive)
        f = F.normalize(feats, dim=-1)
        # f = F.normalize(feats.detach(), dim=-1)
        sel = torch.gather(f, 1, idx.unsqueeze(-1).expand(-1, -1, f.size(-1)))   # [B,k,D]
        k = sel.size(1)
        if k < 2:
            return sel.new_zeros(())
        sim = sel @ sel.transpose(1, 2)
        off = sim - torch.diag_embed(torch.diagonal(sim, dim1=1, dim2=2))
        return off.sum() / (sim.size(0) * k * (k - 1))

    def smoothness_loss(self, scores):
        if scores.size(1) <= 1:
            return torch.tensor(0.0, device=scores.device)
        return torch.mean((scores[:, 1:, :] - scores[:, :-1, :]) ** 2)

    def representativeness_loss(self, feats, idx):
        """
        Encourage the selected frames to represent
        the overall video content.
        """

        feats = F.normalize(feats, dim=-1)

        # Whole-video representation
        video_repr = feats.mean(dim=1)

        # Selected frame features
        selected = torch.gather(
            feats,
            1,
            idx.unsqueeze(-1).expand(
                -1,
                -1,
                feats.size(-1)
            )
        )

        # Summary representation
        summary_repr = selected.mean(dim=1)

        # Cosine similarity
        similarity = F.cosine_similarity(
            summary_repr,
            video_repr,
            dim=-1
        )

        return (1.0 - similarity).mean()

    def total_loss(self, outputs):
        rec = self.reconstruction_loss(outputs['target'], outputs['reconstructed_features'])
        div = self.diversity_loss(outputs['fused_features'], outputs['selected_idx'])
        smooth = self.smoothness_loss(outputs['scores'])
        rep = self.representativeness_loss(outputs['fused_features'],outputs['selected_idx'])
        #rep = self.representativeness_loss(outputs['fused_features'], outputs['mask'])
        total = (self.config.lambda_recon * rec
                + self.config.lambda_div * div
                + self.config.lambda_smooth * smooth
                 + self.config.lambda_rep * rep
                )
        return total, {'total': total, 'reconstruction': rec, 'diversity_loss': div,
                    'smoothness': smooth, 'representativeness_loss': rep
                    }


    
    # def sparsity_loss(self, scores):
    #     return torch.abs(scores.mean() - self.config.summary_rate)

    # def divergence_loss(self, redundancy_r_t):
    #     """Encourages lower state redundancy (higher trajectory divergence)."""
    #     return redundancy_r_t.mean()



        return total, {
            'total': total,
            'reconstruction': reconstruction,
            'sparsity': sparsity
            # 'divergence': divergence,
            # 'smoothness': smoothness
        }

    def diagnose_scores(self, outputs, video_name, epoch_i):
        scores = outputs['scores'].detach()
        p_t = outputs['delta_score'].detach()
        r_t = outputs['state_score'].detach()

        def stats(name, x):
            x = x.float()

            print(
                f"{name:12s} | "
                f"mean={x.mean().item():.6f} | "
                f"std={x.std().item():.6f} | "
                f"min={x.min().item():.6f} | "
                f"max={x.max().item():.6f}"
            )

        print(f"\n===== SCORE DIAGNOSTICS: {video_name} =====")
        d_t = outputs['d_t'].detach()
        delta_t = outputs['delta_t'].detach()
        stats("delta_t", delta_t)
        stats("d_t", d_t)
        stats("p_t", p_t)
        stats("r_t", r_t)
        stats("scores", scores)

        score_1d = scores.squeeze(-1).squeeze(0)

        k = min(15, score_1d.numel())

        top_values, top_indices = torch.topk(
            score_1d,
            k=k
        )

        print("\nTop frame indices:")
        print(top_indices.cpu().tolist())

        print("\nTop frame scores:")
        print(top_values.cpu().tolist())

        print("============================================\n")


    def selection_jaccard(self, current_selected, previous_selected):
        """
        Compute mean Top-K Jaccard similarity between consecutive epochs.

        current_selected / previous_selected:
            dict mapping video_name -> selected frame indices
        """

        values = []
        common_videos = (set(current_selected.keys()) & set(previous_selected.keys()))

        for video_name in common_videos:
            current_set = set(current_selected[video_name])
            previous_set = set(previous_selected[video_name])
            union = current_set | previous_set
            intersection = current_set & previous_set

            if len(union) == 0:
                values.append(1.0)
            else:
                values.append(len(intersection) / len(union))

        if not values:
            return float('nan')
        return sum(values) / len(values)
    

    def train(self):
        for epoch_i in trange(self.config.n_epochs, desc='Epoch'):
            self.summarizer.train()
            epoch_losses = {
                'total': 0.0,
                'reconstruction': 0.0,
                # 'sparsity': 0.0
                # 'divergence': 0.0,
                'smoothness': 0.0,
                'diversity_loss': 0.0,
                'representativeness_loss' : 0.0
            }

            epoch_selected = {}

            # print("Train loaderrrrrrrrr", self.train_loader)
            for batch in tqdm(self.train_loader, desc='Batch', leave=False):
                cnn_features = batch[0].to(self.device)
                semantic_features = batch[1].to(self.device)
                video_name = batch[2][0]
                print(f"\nProcessing video: {video_name}")
                # print("BATCHHHHHHH", batch)
                outputs = self.summarizer(cnn_features, semantic_features)
                
                selected_idx = outputs['selected_idx']
                epoch_selected[video_name] = (
                    selected_idx[0]
                    .detach()
                    .cpu()
                    .tolist()
                )

                total_loss, losses = self.total_loss(outputs)
                # self.diagnose_scores(
                #     outputs,
                #     video_name,
                #     epoch_i
                # )
                
                #reconstruction_target = outputs['fused_features']

                # total_loss, losses = self.total_loss(
                #     reconstruction_target, outputs, epoch_i
                # )
                print("Totall losssss", total_loss)
                self.optimizer.zero_grad()
                total_loss.backward()

                # --- gradient norm logging (before clipping) ---
                def grad_norm(params):
                    return sum(p.grad.norm() ** 2 for p in params if p.grad is not None) ** 0.5

                g_scorer = grad_norm(self.summarizer.interceptor_scorer.parameters())
                g_decoder = grad_norm(self.summarizer.decoder.parameters())
                g_encoder = grad_norm(self.summarizer.mamba_encoder.parameters())
                if self.config.verbose:
                    print(f"GradNorm | scorer={float(g_scorer):.2e} | "
                          f"encoder={float(g_encoder):.2e} | decoder={float(g_decoder):.2e}")
                # -----------------------------------------------

                torch.nn.utils.clip_grad_norm_(
                    self.summarizer.parameters(), self.config.clip
                )
                self.optimizer.step()

                for key in epoch_losses:
                    epoch_losses[key] += losses[key].detach().item()

                if self.config.verbose:
                    scores = outputs["scores"].detach()
                    score_mean = scores.mean().item()
                    score_std = scores.std().item()
                    score_min = scores.min().item()
                    score_max = scores.max().item()

                    # Top 15% frames
                    T = scores.size(1)
                    k = max(1, int(T * self.config.summary_rate))

                    top_scores, _ = torch.topk(scores.squeeze(-1),k=k,dim=1)
                    top15_mean = top_scores.mean().item()
                    fraction_above_05 = (
                        (scores > 0.5).float().mean().item()
                    )
                    print('\nEpoch:', epoch_i)
                    print('Total:', losses['total'].item())
                    print('Reconstruction:', losses['reconstruction'].item())
                    print('Diversity:', losses['diversity_loss'].item())
                    print('Representativeness:', losses['representativeness_loss'].item())
                    print('Smoothness:', losses['smoothness'].item())

                    print(
                    f"Mean={score_mean:.4f} | "
                    f"Std={score_std:.4f} | "
                    f"Min={score_min:.4f} | "
                    f"Max={score_max:.4f} | "
                    f"Top15={top15_mean:.4f} | "
                    f">0.5={fraction_above_05:.4f}"
                    )

            #OLDDDDDDDDDDDDDDDDD
            # num_batches = max(1, len(self.train_loader))
            # print(f'\n================ Epoch {epoch_i} Completed ================')
            # for key in epoch_losses:
            #     print(f'{key}: {epoch_losses[key] / num_batches:.4f}')
            # print('========================================================')
            # ckpt = self.save_checkpoint(epoch_i)
            # self.run_test_evaluate(epoch_i, ckpt)
            #OLDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDD

            num_batches = max(1, len(self.train_loader))
            # Average epoch losses
            avg_losses = {
                key: epoch_losses[key] / num_batches
                for key in epoch_losses
            }

            # Top-K Jaccard against previous epoch
            if self.previous_selected is None:
                topk_jaccard = float('nan')
            else:
                topk_jaccard = self.selection_jaccard(epoch_selected,self.previous_selected)
            # Store current selections for next epoch
            self.previous_selected = {
                video_name: indices.copy()
                for video_name, indices in epoch_selected.items()
            }

            # Save losses + Top-K Jaccard
            with open(self.metrics_csv, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    self.config.split_index,
                    epoch_i,
                    avg_losses['total'],
                    avg_losses['reconstruction'],
                    avg_losses['diversity_loss'],
                    avg_losses['representativeness_loss'],
                    avg_losses['smoothness'],
                    topk_jaccard
                ])

            print(f'\n================ Epoch {epoch_i} Completed ================')

            for key in avg_losses:
                print(f'{key}: {avg_losses[key]:.4f}')

            print('========================================================')
            ckpt = self.save_checkpoint(epoch_i)
            self.run_test_evaluate(epoch_i, ckpt)

        # self.evaluate(epoch_i)
        # self.save_checkpoint(epoch_i)

    def evaluate(self, epoch_i):
        self.summarizer.eval()
        results = {}

        with torch.no_grad():
            for batch in tqdm(self.test_loader, desc='Evaluate', leave=False):
                cnn_features = batch[0].to(self.device)
                semantic_features = batch[1].to(self.device)
                video_name = batch[2]

                outputs = self.summarizer(cnn_features, semantic_features)
                scores = outputs['scores'].squeeze(-1)[0].cpu().numpy().tolist()
                results[video_name] = scores

        self.config.score_dir.mkdir(parents=True, exist_ok=True)
        output_file = self.config.score_dir / f'{self.config.video_type}_epoch_{epoch_i}.json'

        with open(output_file, 'w') as f:
            json.dump(results, f)

        print('Saved scores to:', output_file)

    def run_test_evaluate(self, epoch_i, ckpt):
        out_dir = Path('./summe/test'); out_dir.mkdir(parents=True, exist_ok=True)
        out_h5 = out_dir / 'result_test.h5'
        cmd = [sys.executable, 'Test_evaluate.py',
               '--dataset', 'summe',
               '--cnn-h5', './datasets/summe/eccv16_dataset_summe_google_pool5.h5',
               '--semantic-h5', './datasets/summe/eccv16_dataset_summe_siglip2.h5',
               '--checkpoint', str(ckpt),
               '--output', str(out_h5),
               '--splits-file', './datasets/summe/splits/summe_splits.json',
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

        csv_path = out_dir / f'epoch_f1_epoch_{self.config.n_epochs}_split_{self.config.split_index}_temporal_wgt_updt_ver3_topk.csv'
        new_file = not csv_path.exists()
        with open(csv_path, 'a', newline='') as f:
            w = csv.writer(f)
            if new_file:
                w.writerow(['split', 'epoch', 'f1', 'precision', 'recall'])
            w.writerow([self.config.split_index, epoch_i, f1, precision, recall])
        print(f'[eval] epoch {epoch_i}: F1 = {f1}')
        # Save a separate copy of the best checkpoint (only after epoch 10)
        if isinstance(f1, (int, float)) and epoch_i > 10 and f1 > self.best_f1:
            self.best_f1, self.best_epoch = f1, epoch_i
            best_path = Path(ckpt).with_name(Path(ckpt).stem + '_best_f1.pth')
            shutil.copyfile(ckpt, best_path)
            print(f'[eval] new best F1 = {f1} at epoch {epoch_i} -> {best_path}')

    def save_checkpoint(self, epoch_i):
        self.config.save_dir.mkdir(parents=True, exist_ok=True)

        checkpoint_path = self.config.save_dir / f'{self.config.video_type}_epoch_{self.config.n_epochs}_cd_wSmoothReptDivRwd_temporal_wgt_updt_ver3_topk.pth'
        
        # checkpoint_path = self.config.save_dir / f'{self.config.video_type}_epoch_{epoch_i}_newLayer_cd_wSmoothReptCvRwd.pth'

        torch.save({
            'epoch': epoch_i,
            'model_state_dict': self.summarizer.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict()
        }, checkpoint_path)
        print('Saved checkpoint:', checkpoint_path)
        return checkpoint_path
# -*- coding: utf-8 -*-

import json
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm, trange
from Summarizer import Summarizer


class Solver:
    def __init__(self, config, train_loader, test_loader):
        self.config = config
        self.train_loader = train_loader
        self.test_loader = test_loader

    def build(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print('Using device:', self.device)

        self.summarizer = Summarizer(
            cnn_size=self.config.input_size,
            semantic_size=self.config.semantic_size,
            hidden_size=self.config.hidden_size,
            d_state=self.config.d_state,
            dropout=self.config.dropout
        ).to(self.device)

        self.optimizer = optim.AdamW(
            self.summarizer.parameters(),
            lr=self.config.lr,
            weight_decay=self.config.weight_decay
        )

        total_parameters = sum(
            p.numel() for p in self.summarizer.parameters() if p.requires_grad
        )
        print('Trainable parameters:', total_parameters)

    # Losses
    def reconstruction_loss(self, target, reconstructed):
        return torch.mean(torch.norm(target - reconstructed, p=2, dim=-1))

    def sparsity_loss(self, scores):
        return torch.abs(scores.mean() - self.config.summary_rate)

    def divergence_loss(self, redundancy_r_t):
        """Encourages lower state redundancy (higher trajectory divergence)."""
        return redundancy_r_t.mean()

    def smoothness_loss(self, scores):
        if scores.size(1) <= 1:
            return torch.tensor(0.0, device=scores.device)
        return torch.mean((scores[:, 1:, :] - scores[:, :-1, :]) ** 2)

    def total_loss(self, target, outputs, epoch_i):
        reconstruction = self.reconstruction_loss(target, outputs['reconstructed_features'])
        sparsity = self.sparsity_loss(outputs['scores'])
        # divergence = self.divergence_loss(outputs['redundancy_r_t'])
        # smoothness = self.smoothness_loss(outputs['scores'])

        # Dynamic Sparsity Loss Warmup Schedule
        warmup_epochs = self.config.warmup_epochs
        sparse_weight = self.config.lambda_sparse * min(1.0, (epoch_i + 1) / warmup_epochs)

        total = (
            self.config.lambda_recon * reconstruction
            + sparse_weight * sparsity
            # + self.config.lambda_div * divergence
            # + self.config.lambda_smooth * smoothness
        )

        return total, {
            'total': total,
            'reconstruction': reconstruction,
            'sparsity': sparsity
            # 'divergence': divergence,
            # 'smoothness': smoothness
        }

    def diagnose_scores(self, outputs, video_name, epoch_i):
        scores = outputs['scores'].detach()
        p_t = outputs['novelty_p_t'].detach()
        r_t = outputs['redundancy_r_t'].detach()

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

    def train(self):
        for epoch_i in trange(self.config.n_epochs, desc='Epoch'):
            self.summarizer.train()
            epoch_losses = {
                'total': 0.0,
                'reconstruction': 0.0,
                'sparsity': 0.0
                # 'divergence': 0.0,
                # 'smoothness': 0.0
            }
            # print("Train loaderrrrrrrrr", self.train_loader)
            for batch in tqdm(self.train_loader, desc='Batch', leave=False):
                cnn_features = batch[0].to(self.device)
                semantic_features = batch[1].to(self.device)
                video_name = batch[2][0]
                # print(f"\nProcessing video: {video_name}")
                # print("BATCHHHHHHH", batch)
                outputs = self.summarizer(cnn_features, semantic_features)
                
                # self.diagnose_scores(
                #     outputs,
                #     video_name,
                #     epoch_i
                # )
                
                reconstruction_target = outputs['fused_features']

                total_loss, losses = self.total_loss(
                    reconstruction_target, outputs, epoch_i
                )
                print("Totall losssss", total_loss)
                self.optimizer.zero_grad()
                total_loss.backward()

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
                    print(
                    f"Mean={score_mean:.4f} | "
                    f"Std={score_std:.4f} | "
                    f"Min={score_min:.4f} | "
                    f"Max={score_max:.4f} | "
                    f"Top15={top15_mean:.4f} | "
                    f">0.5={fraction_above_05:.4f}"
                    )

            num_batches = max(1, len(self.train_loader))
            print(f'\n================ Epoch {epoch_i} Completed ================')
            for key in epoch_losses:
                print(f'{key}: {epoch_losses[key] / num_batches:.4f}')
            print('========================================================')

        self.evaluate(epoch_i)
        self.save_checkpoint(epoch_i)

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

    def save_checkpoint(self, epoch_i):
        self.config.save_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_path = self.config.save_dir / f'{self.config.video_type}_epoch_{epoch_i}_newLayer.pth'

        torch.save({
            'epoch': epoch_i,
            'model_state_dict': self.summarizer.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict()
        }, checkpoint_path)
        print('Saved checkpoint:', checkpoint_path)
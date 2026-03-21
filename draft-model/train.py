#!/usr/bin/env python3
"""
Train the SRAM-resident draft model on financial corpus.

Phase 1: Cross-entropy on hard labels (next-token prediction).
Uses PyTorch on MPS (Apple GPU) for training, then exports for CoreML.

~40M parameters, 4.1M training tokens, should train in 30-60 minutes on M5.
"""

import os
import sys
import time
import json
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# Add model directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model import DraftModel

TRAIN_DATA_DIR = "/Users/midas/Desktop/cowork/ane-perf/draft-model/train_data"
VOCAB_MAP_PATH = "/Users/midas/Desktop/cowork/ane-perf/draft-model/vocab_map.json"
OUTPUT_DIR = "/Users/midas/Desktop/cowork/ane-perf/draft-model/checkpoints"

# Training hyperparameters
BATCH_SIZE = 8
LEARNING_RATE = 3e-4
WEIGHT_DECAY = 0.1
WARMUP_STEPS = 200
MAX_STEPS = 5000
EVAL_EVERY = 250
SAVE_EVERY = 1000
GRAD_CLIP = 1.0


class TokenDataset(Dataset):
    def __init__(self, input_ids, target_ids):
        self.input_ids = torch.from_numpy(input_ids).long()
        self.target_ids = torch.from_numpy(target_ids).long()

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return self.input_ids[idx], self.target_ids[idx]


def get_lr(step, warmup_steps, max_lr, max_steps):
    """Cosine schedule with linear warmup."""
    if step < warmup_steps:
        return max_lr * step / warmup_steps
    progress = (step - warmup_steps) / (max_steps - warmup_steps)
    return max_lr * 0.5 * (1.0 + math.cos(math.pi * progress))


def train():
    # Load data
    print("Loading training data...")
    input_ids = np.load(os.path.join(TRAIN_DATA_DIR, "input_ids.npy"))
    target_ids = np.load(os.path.join(TRAIN_DATA_DIR, "target_ids.npy"))

    # Split 95/5 train/val
    n = len(input_ids)
    n_val = max(100, n // 20)
    n_train = n - n_val
    perm = np.random.RandomState(42).permutation(n)
    train_idx = perm[:n_train]
    val_idx = perm[n_train:]

    train_ds = TokenDataset(input_ids[train_idx], target_ids[train_idx])
    val_ds = TokenDataset(input_ids[val_idx], target_ids[val_idx])

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              drop_last=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False,
                            num_workers=0)

    print(f"Train: {n_train} chunks, Val: {n_val} chunks")
    print(f"Steps per epoch: {len(train_loader)}")

    # Load vocab info
    with open(VOCAB_MAP_PATH) as f:
        vmap = json.load(f)
    vocab_size = vmap["pruned_vocab_size"]

    # Model
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Device: {device}")

    model = DraftModel(vocab_size=vocab_size).to(device)
    params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {params:,} ({params/1e6:.1f}M)")

    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY,
        betas=(0.9, 0.95)
    )

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Training loop
    train_iter = iter(train_loader)
    best_val_loss = float("inf")
    t0 = time.time()
    running_loss = 0.0
    running_count = 0

    print(f"\nTraining for {MAX_STEPS} steps...")
    print(f"{'Step':>6} {'Loss':>8} {'LR':>10} {'tok/s':>8} {'Time':>8}")
    print("-" * 50)

    for step in range(1, MAX_STEPS + 1):
        # Get batch (cycle through epochs)
        try:
            batch_x, batch_y = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            batch_x, batch_y = next(train_iter)

        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)

        # LR schedule
        lr = get_lr(step, WARMUP_STEPS, LEARNING_RATE, MAX_STEPS)
        for pg in optimizer.param_groups:
            pg["lr"] = lr

        # Forward
        logits = model(batch_x)  # (B, T, vocab)
        loss = F.cross_entropy(
            logits.view(-1, vocab_size),
            batch_y.view(-1),
            ignore_index=0  # ignore UNK/padding
        )

        # Backward
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        optimizer.step()

        running_loss += loss.item()
        running_count += 1

        # Log
        if step % 50 == 0:
            avg_loss = running_loss / running_count
            elapsed = time.time() - t0
            tokens = step * BATCH_SIZE * 511
            tok_s = tokens / elapsed
            print(f"{step:6d} {avg_loss:8.4f} {lr:10.6f} {tok_s:8.0f} {elapsed:7.0f}s")
            running_loss = 0.0
            running_count = 0

        # Eval
        if step % EVAL_EVERY == 0:
            model.eval()
            val_loss = 0.0
            val_tokens = 0
            with torch.no_grad():
                for vx, vy in val_loader:
                    vx, vy = vx.to(device), vy.to(device)
                    logits = model(vx)
                    loss = F.cross_entropy(
                        logits.view(-1, vocab_size), vy.view(-1),
                        ignore_index=0, reduction="sum"
                    )
                    val_loss += loss.item()
                    val_tokens += (vy != 0).sum().item()
            val_loss /= val_tokens
            ppl = math.exp(min(val_loss, 20))
            print(f"  >> Val loss: {val_loss:.4f}, Perplexity: {ppl:.1f}")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(model.state_dict(),
                           os.path.join(OUTPUT_DIR, "best.pt"))
                print(f"  >> New best! Saved to {OUTPUT_DIR}/best.pt")

            model.train()

        # Periodic save
        if step % SAVE_EVERY == 0:
            torch.save({
                "step": step,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": best_val_loss,
            }, os.path.join(OUTPUT_DIR, f"step_{step}.pt"))

    # Final save
    torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, "final.pt"))
    elapsed = time.time() - t0
    print(f"\nDone. {MAX_STEPS} steps in {elapsed/60:.1f} minutes")
    print(f"Best val loss: {best_val_loss:.4f} (ppl {math.exp(min(best_val_loss, 20)):.1f})")
    print(f"Saved: {OUTPUT_DIR}/best.pt, {OUTPUT_DIR}/final.pt")


if __name__ == "__main__":
    train()

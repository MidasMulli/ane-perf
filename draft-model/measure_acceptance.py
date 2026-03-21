#!/usr/bin/env python3
"""
Measure draft model acceptance rate against Qwen3.5-9B.

For each position in test prompts:
  1. Run draft model, get top prediction (in pruned vocab)
  2. Map to full vocab
  3. Run 9B, get top prediction
  4. Count matches

Acceptance rate = matches / total positions
"""

import json
import os
import sys
import time
import numpy as np
import torch

assert os.path.basename(os.getcwd()) != "ane-perf", \
    "Run from /tmp to avoid profile.py shadowing stdlib"

sys.path.insert(0, "/Users/midas/Desktop/cowork/ane-perf/draft-model")
from model import DraftModel

import mlx.core as mx
from mlx_lm import load as mlx_load
from transformers import AutoTokenizer

CHECKPOINT = "/Users/midas/Desktop/cowork/ane-perf/draft-model/checkpoints/best.pt"
VOCAB_MAP = "/Users/midas/Desktop/cowork/ane-perf/draft-model/vocab_map.json"
TEACHER_MODEL = "mlx-community/Qwen3.5-9B-MLX-4bit"

TEST_PROMPTS = [
    # ISDA / financial
    "The ISDA Master Agreement governs the relationship between counterparties in over-the-counter derivative transactions. Under Section 6, either party may designate an Early Termination Date if an Event of Default has occurred with respect to the other party. The Close-out Amount shall be determined by the",
    "Credit Support Annex paragraph 3 specifies that the Valuation Agent shall calculate the Credit Support Amount on each Valuation Date. If the Delivery Amount exceeds the Minimum Transfer Amount, the Pledgor shall",
    "For purposes of calculating the firm's regulatory capital ratios under the Basel III standardized approach, risk-weighted assets are determined by multiplying the exposure amount of each on-balance sheet asset by the",
    "Total return swaps on investment grade credit indices provide synthetic exposure to corporate bond markets. The total return receiver pays SOFR plus a spread and receives the total return including",
    # Analytical / general financial
    "JPMorgan Chase reported net revenue of $42.4 billion for the quarter, driven by higher net interest income and strong performance in the investment banking division. Management noted that",
    "The company's liquidity coverage ratio remained well above regulatory minimums at 112%, reflecting a diversified funding base and a high-quality liquid asset portfolio consisting primarily of",
]


def main():
    # Load vocab map
    with open(VOCAB_MAP) as f:
        vmap = json.load(f)
    full_to_pruned = {int(k): v for k, v in vmap["full_to_pruned"].items()}
    pruned_to_full = {int(k): v for k, v in vmap["pruned_to_full"].items()}
    pruned_vocab_size = vmap["pruned_vocab_size"]

    # Load draft model
    print("Loading draft model...")
    draft = DraftModel(vocab_size=pruned_vocab_size)
    state = torch.load(CHECKPOINT, map_location="cpu", weights_only=True)
    draft.load_state_dict(state)
    draft.eval()

    # Load teacher (9B)
    print("Loading teacher (9B)...")
    teacher, tokenizer = mlx_load(TEACHER_MODEL)

    print(f"\nMeasuring acceptance on {len(TEST_PROMPTS)} prompts...")
    print("=" * 70)

    total_match = 0
    total_pos = 0
    gen_length = 128  # positions to test per prompt

    for pi, prompt in enumerate(TEST_PROMPTS):
        tokens = tokenizer.encode(prompt)
        matches = 0
        positions = 0

        # Autoregressive: generate one token at a time, compare draft vs teacher
        current_tokens = list(tokens)

        for pos in range(gen_length):
            # Draft prediction
            draft_input = [full_to_pruned.get(t, 0) for t in current_tokens]
            with torch.no_grad():
                draft_logits = draft(torch.tensor([draft_input]))
                draft_pred_pruned = draft_logits[0, -1].argmax().item()
            draft_pred_full = pruned_to_full.get(draft_pred_pruned, 0)

            # Teacher prediction
            teacher_input = mx.array([current_tokens])
            teacher_logits = teacher(teacher_input)
            mx.eval(teacher_logits)
            teacher_pred = int(teacher_logits[0, -1].argmax())

            if draft_pred_full == teacher_pred:
                matches += 1

            # Advance with teacher's token (oracle feeding)
            current_tokens.append(teacher_pred)
            positions += 1

        rate = matches / positions * 100
        total_match += matches
        total_pos += positions

        prompt_short = prompt[:60] + "..."
        print(f"\n  Prompt {pi+1}: {prompt_short}")
        print(f"  Acceptance: {matches}/{positions} ({rate:.1f}%)")

        # Show first few predictions
        print(f"  Sample predictions (draft -> teacher):")
        current_tokens = list(tokens)
        for j in range(min(10, gen_length)):
            draft_input = [full_to_pruned.get(t, 0) for t in current_tokens]
            with torch.no_grad():
                dl = draft(torch.tensor([draft_input]))
                dp = pruned_to_full.get(dl[0, -1].argmax().item(), 0)
            ti = mx.array([current_tokens])
            tl = teacher(ti)
            mx.eval(tl)
            tp = int(tl[0, -1].argmax())
            match = "Y" if dp == tp else " "
            current_tokens.append(tp)
            print(f"    [{match}] draft={tokenizer.decode([dp])!r:20s} "
                  f"teacher={tokenizer.decode([tp])!r}")

    overall = total_match / total_pos * 100
    print(f"\n{'=' * 70}")
    print(f"Overall acceptance: {total_match}/{total_pos} ({overall:.1f}%)")
    print(f"\nVerdict: {'VIABLE (>10%)' if overall > 10 else 'TOO LOW (<10%)'}")

    if overall > 10:
        print("  -> Wire into four-path server as ANE draft source")
    elif overall > 5:
        print("  -> Marginal. Try KL distillation (Phase 2) before giving up")
    else:
        print("  -> Model too small or not enough data. Consider dim=1536")


if __name__ == "__main__":
    main()

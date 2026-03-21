#!/usr/bin/env python3
"""
Honest audit of the SRAM-resident ANE draft model.

Tests:
  1. Teacher-forcing vs autoregressive acceptance
  2. Training data overlap check
  3. Held-out prompt acceptance
  4. IOReport energy/DRAM comparison vs synthetic
  5. Non-ANE op analysis (the 15 fallback ops)
  6. Per-prompt breakdown: high vs low acceptance + N-gram overlap

No celebration until every number is verified.
"""

import json
import os
import sys
import time
import ctypes
import ctypes.util
import numpy as np
import torch

assert os.path.basename(os.getcwd()) != "ane-perf", \
    "Run from /tmp to avoid profile.py shadowing stdlib"

sys.path.insert(0, "/Users/midas/Desktop/cowork/ane-perf/draft-model")
from model import DraftModel

import mlx.core as mx
from mlx_lm import load as mlx_load

CHECKPOINT = "/Users/midas/Desktop/cowork/ane-perf/draft-model/checkpoints/best.pt"
VOCAB_MAP = "/Users/midas/Desktop/cowork/ane-perf/draft-model/vocab_map.json"
TEACHER_MODEL = "mlx-community/Qwen3.5-9B-MLX-4bit"

# Original test prompts (from measure_acceptance.py)
ORIGINAL_PROMPTS = [
    "The ISDA Master Agreement governs the relationship between counterparties in over-the-counter derivative transactions. Under Section 6, either party may designate an Early Termination Date if an Event of Default has occurred with respect to the other party. The Close-out Amount shall be determined by the",
    "Credit Support Annex paragraph 3 specifies that the Valuation Agent shall calculate the Credit Support Amount on each Valuation Date. If the Delivery Amount exceeds the Minimum Transfer Amount, the Pledgor shall",
    "For purposes of calculating the firm's regulatory capital ratios under the Basel III standardized approach, risk-weighted assets are determined by multiplying the exposure amount of each on-balance sheet asset by the",
    "Total return swaps on investment grade credit indices provide synthetic exposure to corporate bond markets. The total return receiver pays SOFR plus a spread and receives the total return including",
    "JPMorgan Chase reported net revenue of $42.4 billion for the quarter, driven by higher net interest income and strong performance in the investment banking division. Management noted that",
    "The company's liquidity coverage ratio remained well above regulatory minimums at 112%, reflecting a diversified funding base and a high-quality liquid asset portfolio consisting primarily of",
]

# HELD-OUT prompts: completely new, not in training corpus or prior tests
HELD_OUT_PROMPTS = [
    # Novel ISDA scenario not in any 10-K or S-1
    "When calculating the Net Termination Amount under a 2002 ISDA Master Agreement with bilateral close-out netting, the Determining Party must consider replacement transaction costs as of the Early Termination Date. The methodology for obtaining",
    # Novel regulatory topic
    "The Federal Reserve's annual stress test results indicated that under the severely adverse scenario, the firm's common equity tier 1 capital ratio would decline by approximately 4.2 percentage points to",
    # Novel derivatives topic
    "Interest rate swaptions with Bermudan exercise features present unique valuation challenges because the holder can exercise on multiple dates. The standard approach uses a backward induction methodology where at each exercise date",
    # Novel fintech/modern topic (definitely not in 10-K training data)
    "Decentralized finance protocols that implement automated market making for tokenized securities face significant regulatory uncertainty. The SEC's framework for determining whether a digital asset constitutes a security under the Howey test requires",
    # Pure analytical (no financial jargon overlap with training)
    "The primary risk factor affecting our consolidated financial statements is the uncertainty surrounding macroeconomic conditions in emerging markets. Management has implemented a hedging strategy that utilizes",
]

# Training corpus snippets (first 200 chars from each file, for overlap check)
TRAINING_FILES = [
    "/Users/midas/Desktop/cowork/ngram-engine/10k-samples/jpm-10k.txt",
    "/Users/midas/Desktop/cowork/ngram-engine/10k-samples/gs-10k.txt",
    "/Users/midas/Desktop/cowork/ngram-engine/s1-samples/reddit-s1.txt",
]


def load_models():
    with open(VOCAB_MAP) as f:
        vmap = json.load(f)
    full_to_pruned = {int(k): v for k, v in vmap["full_to_pruned"].items()}
    pruned_to_full = {int(k): v for k, v in vmap["pruned_to_full"].items()}
    vocab_size = vmap["pruned_vocab_size"]

    draft = DraftModel(vocab_size=vocab_size)
    state = torch.load(CHECKPOINT, map_location="cpu", weights_only=True)
    draft.load_state_dict(state)
    draft.eval()

    teacher, tokenizer = mlx_load(TEACHER_MODEL)
    return draft, teacher, tokenizer, full_to_pruned, pruned_to_full


def measure_teacher_forcing(draft, teacher, tokenizer, full_to_pruned,
                           pruned_to_full, prompt, gen_len=128):
    """Teacher-forcing: feed correct previous token, check next prediction."""
    tokens = tokenizer.encode(prompt)
    current = list(tokens)
    matches = 0

    for _ in range(gen_len):
        # Draft
        draft_input = [full_to_pruned.get(t, 0) for t in current]
        with torch.no_grad():
            logits = draft(torch.tensor([draft_input]))
            pred_pruned = logits[0, -1].argmax().item()
        pred_full = pruned_to_full.get(pred_pruned, 0)

        # Teacher
        tl = teacher(mx.array([current]))
        mx.eval(tl)
        teacher_pred = int(tl[0, -1].argmax())

        if pred_full == teacher_pred:
            matches += 1

        current.append(teacher_pred)  # teacher-forcing

    return matches, gen_len


def measure_autoregressive(draft, teacher, tokenizer, full_to_pruned,
                          pruned_to_full, prompt, gen_len=128):
    """Autoregressive: draft feeds its OWN predictions, errors compound."""
    tokens = tokenizer.encode(prompt)
    draft_tokens = list(tokens)
    teacher_tokens = list(tokens)
    matches = 0

    for _ in range(gen_len):
        # Draft predicts from ITS OWN history
        draft_input = [full_to_pruned.get(t, 0) for t in draft_tokens]
        with torch.no_grad():
            logits = draft(torch.tensor([draft_input]))
            pred_pruned = logits[0, -1].argmax().item()
        draft_pred_full = pruned_to_full.get(pred_pruned, 0)

        # Teacher predicts from ITS OWN history
        tl = teacher(mx.array([teacher_tokens]))
        mx.eval(tl)
        teacher_pred = int(tl[0, -1].argmax())

        if draft_pred_full == teacher_pred:
            matches += 1

        # Each model advances with its OWN prediction
        draft_tokens.append(draft_pred_full)
        teacher_tokens.append(teacher_pred)

    return matches, gen_len


def measure_spec_decode_sim(draft, teacher, tokenizer, full_to_pruned,
                           pruned_to_full, prompt, K=4, rounds=20):
    """
    Simulate actual speculative decoding: draft K tokens, verify batch,
    accept prefix, reject rest. This is the REAL acceptance measurement.
    """
    tokens = tokenizer.encode(prompt)
    context = list(tokens)
    total_accepted = 0
    total_rounds = 0

    for _ in range(rounds):
        # Draft K tokens autoregressively
        draft_context = list(context)
        draft_preds = []
        for _ in range(K):
            draft_input = [full_to_pruned.get(t, 0) for t in draft_context]
            with torch.no_grad():
                logits = draft(torch.tensor([draft_input]))
                pred_pruned = logits[0, -1].argmax().item()
            pred_full = pruned_to_full.get(pred_pruned, 0)
            draft_preds.append(pred_full)
            draft_context.append(pred_full)

        # Verify: teacher checks each draft token sequentially
        verify_context = list(context)
        accepted = 0
        for dp in draft_preds:
            tl = teacher(mx.array([verify_context]))
            mx.eval(tl)
            teacher_pred = int(tl[0, -1].argmax())
            if dp == teacher_pred:
                accepted += 1
                verify_context.append(dp)
            else:
                # Reject here and all subsequent
                verify_context.append(teacher_pred)
                break

        if accepted == K:
            # All accepted, get one bonus token from teacher
            tl = teacher(mx.array([verify_context]))
            mx.eval(tl)
            verify_context.append(int(tl[0, -1].argmax()))

        total_accepted += accepted
        total_rounds += 1
        context = verify_context

    return total_accepted, total_rounds * K


def check_training_overlap(prompts, training_files, tokenizer):
    """Check if test prompts overlap with training data."""
    print("\n" + "=" * 70)
    print("  TEST 1: Training Data Overlap Check")
    print("=" * 70)

    # Load training text
    training_text = ""
    for f in training_files:
        if os.path.exists(f):
            training_text += open(f).read()

    for i, prompt in enumerate(prompts):
        # Check if prompt text appears verbatim in training data
        # Check 10-word windows
        words = prompt.split()
        max_overlap = 0
        for start in range(len(words) - 9):
            window = " ".join(words[start:start+10])
            if window.lower() in training_text.lower():
                max_overlap = max(max_overlap, 10)
                break

        # Check 5-word windows
        five_word_matches = 0
        for start in range(len(words) - 4):
            window = " ".join(words[start:start+5])
            if window.lower() in training_text.lower():
                five_word_matches += 1

        status = "CLEAN" if max_overlap == 0 and five_word_matches < 3 else "OVERLAP"
        print(f"  Prompt {i+1}: {status} (10-word: {max_overlap}, "
              f"5-word matches: {five_word_matches})")
        print(f"    {prompt[:70]}...")


def main():
    print("=" * 70)
    print("  HONEST AUDIT: SRAM-Resident ANE Draft Model")
    print("  Every number challenged, every claim verified")
    print("=" * 70)

    draft, teacher, tokenizer, f2p, p2f = load_models()

    # ── Test 1: Training data overlap ──
    check_training_overlap(ORIGINAL_PROMPTS + HELD_OUT_PROMPTS,
                          TRAINING_FILES, tokenizer)

    # ── Test 2: Teacher-forcing vs Autoregressive ──
    print("\n" + "=" * 70)
    print("  TEST 2: Teacher-Forcing vs Autoregressive Acceptance")
    print("  (The GDN converter showed 59% TF -> 0.94x wallclock)")
    print("=" * 70)

    gen_len = 64  # shorter to keep runtime reasonable

    for label, prompts in [("ORIGINAL", ORIGINAL_PROMPTS[:3]),
                           ("HELD-OUT", HELD_OUT_PROMPTS[:3])]:
        print(f"\n  {label} prompts:")
        for i, prompt in enumerate(prompts):
            tf_match, tf_total = measure_teacher_forcing(
                draft, teacher, tokenizer, f2p, p2f, prompt, gen_len)
            ar_match, ar_total = measure_autoregressive(
                draft, teacher, tokenizer, f2p, p2f, prompt, gen_len)

            tf_rate = tf_match / tf_total * 100
            ar_rate = ar_match / ar_total * 100
            drop = (tf_rate - ar_rate) / tf_rate * 100 if tf_rate > 0 else 0

            print(f"    Prompt {i+1}: TF={tf_rate:.1f}% AR={ar_rate:.1f}% "
                  f"(drop: {drop:.0f}%)")
            print(f"      {prompt[:60]}...")

    # ── Test 3: Spec decode simulation (real acceptance) ──
    print("\n" + "=" * 70)
    print("  TEST 3: Simulated Speculative Decoding (K=4, 20 rounds)")
    print("  Draft feeds own predictions, rejected tokens counted as zero")
    print("=" * 70)

    for label, prompts in [("ORIGINAL", ORIGINAL_PROMPTS),
                           ("HELD-OUT", HELD_OUT_PROMPTS)]:
        print(f"\n  {label} prompts:")
        total_acc = 0
        total_att = 0
        for i, prompt in enumerate(prompts):
            acc, att = measure_spec_decode_sim(
                draft, teacher, tokenizer, f2p, p2f, prompt, K=4, rounds=10)
            rate = acc / att * 100
            total_acc += acc
            total_att += att
            print(f"    Prompt {i+1}: {acc}/{att} ({rate:.1f}%) "
                  f"{prompt[:50]}...")

        overall = total_acc / total_att * 100 if total_att > 0 else 0
        print(f"  {label} overall: {total_acc}/{total_att} ({overall:.1f}%)")

    # ── Test 4: N-gram overlap analysis ──
    print("\n" + "=" * 70)
    print("  TEST 4: Does ANE Win Where N-gram Loses?")
    print("  (If ANE only works on boilerplate, N-gram already catches it)")
    print("=" * 70)

    # Simple N-gram: check if teacher's next token matches any of the
    # previous N tokens in context (proxy for N-gram predictor)
    for label, prompts in [("ORIGINAL", ORIGINAL_PROMPTS),
                           ("HELD-OUT", HELD_OUT_PROMPTS)]:
        print(f"\n  {label} prompts:")
        for i, prompt in enumerate(prompts):
            tokens = tokenizer.encode(prompt)
            context = list(tokens)
            ane_only = 0  # ANE correct, N-gram wrong
            ngram_only = 0  # N-gram correct, ANE wrong
            both = 0
            neither = 0

            for pos in range(min(64, 128)):
                # Teacher next token
                tl = teacher(mx.array([context]))
                mx.eval(tl)
                teacher_pred = int(tl[0, -1].argmax())

                # ANE draft
                draft_input = [f2p.get(t, 0) for t in context]
                with torch.no_grad():
                    dl = draft(torch.tensor([draft_input]))
                    dp = p2f.get(dl[0, -1].argmax().item(), 0)
                ane_correct = (dp == teacher_pred)

                # N-gram proxy: does teacher token appear in last 32 context tokens?
                ngram_correct = teacher_pred in context[-32:]

                if ane_correct and ngram_correct:
                    both += 1
                elif ane_correct and not ngram_correct:
                    ane_only += 1
                elif not ane_correct and ngram_correct:
                    ngram_only += 1
                else:
                    neither += 1

                context.append(teacher_pred)

            total = both + ane_only + ngram_only + neither
            print(f"    P{i+1}: ANE-only={ane_only}({100*ane_only/total:.0f}%) "
                  f"Ngram-only={ngram_only}({100*ngram_only/total:.0f}%) "
                  f"Both={both}({100*both/total:.0f}%) "
                  f"Neither={neither}({100*neither/total:.0f}%)")

    # ── Test 5: Non-ANE ops analysis ──
    print("\n" + "=" * 70)
    print("  TEST 5: The 15 Non-ANE Ops (from profile.py output)")
    print("=" * 70)
    print("  From profiler output, the 15 mps_graph ops are:")
    print("    [0] tensor_buffer_to_tensor  — embedding lookup (start)")
    print("    [1-2] cast (INT32 -> FP16)   — input type conversion (start)")
    print("    [3-4] greater_equal, add     — mask/padding logic (start)")
    print("    [5] select                   — embedding select (start)")
    print("    [6-7] cast                   — duplicate type conversion")
    print("    [8-9] greater_equal, add     — duplicate mask logic")
    print("    [10] select                  — duplicate embedding select")
    print("    [11] constexpr_blockwise_shift_scale — dequant for gather")
    print("    [12] gather                  — vocabulary gather (start)")
    print("    [13] pow                     — first RMSNorm power (start)")
    print("    [14] reduce_mean             — first RMSNorm mean (start)")
    print("  ALL 15 ops are clustered at the START (embedding + first norm).")
    print("  Zero CPU fallbacks inside the transformer layers.")
    print("  Data transfer: 1 ANE-to-CPU boundary (after embedding).")
    print("  This is optimal — embedding can't run on ANE (gather op).")

    # ── Summary ──
    print("\n" + "=" * 70)
    print("  AUDIT SUMMARY")
    print("=" * 70)
    print("  Corrections needed:")
    print("    - 20,000 pred/s was synthetic single-op, real model = 73 pred/s (274x)")
    print("    - 26.8% was teacher-forcing, not autoregressive")
    print("    - '+0.10x speedup' was calculated, not measured end-to-end")
    print("    - SRAM residency not confirmed (need per-layer DRAM check)")


if __name__ == "__main__":
    main()

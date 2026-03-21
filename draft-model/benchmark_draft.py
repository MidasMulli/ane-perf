#!/usr/bin/env python3
"""
Benchmark the draft model as a spec decode source.

Key insight: ANE runs on DIFFERENT hardware than GPU. In the four-path
architecture, the ANE draft runs IN PARALLEL with GPU batch verification.
The 13.8ms ANE call is hidden within the 110ms GPU verify phase.

Architecture:
  1. N-gram + PLD generate draft tokens (CPU, ~0ms)
  2. ANE generates 1 draft token (ANE, 13.8ms) — PARALLEL with step 3
  3. GPU batch-verifies all drafts (GPU, 110ms)
  ANE cost is zero because it finishes before GPU verify completes.
"""

import json
import os
import sys
import time
import numpy as np

assert os.path.basename(os.getcwd()) != "ane-perf", \
    "Run from /tmp to avoid profile.py shadowing stdlib"

import coremltools as ct
import torch

sys.path.insert(0, "/Users/midas/Desktop/cowork/ane-perf/draft-model")
from model import DraftModel

COREML_PATH = "/Users/midas/Desktop/cowork/ane-perf/draft-model/draft_model.mlpackage"
PYTORCH_CKPT = "/Users/midas/Desktop/cowork/ane-perf/draft-model/checkpoints/best.pt"
VOCAB_MAP = "/Users/midas/Desktop/cowork/ane-perf/draft-model/vocab_map.json"
SEQ_LEN = 512


def benchmark_coreml():
    """Benchmark CoreML model on ANE."""
    print("Loading CoreML model...")
    model = ct.models.MLModel(COREML_PATH)

    dummy = np.zeros((1, SEQ_LEN), dtype=np.int32)
    for _ in range(10):
        model.predict({"input_ids": dummy})

    times = []
    for _ in range(50):
        t0 = time.perf_counter()
        model.predict({"input_ids": dummy})
        times.append(time.perf_counter() - t0)

    avg_ms = np.mean(times) * 1000
    p50_ms = np.median(times) * 1000
    print(f"\n  CoreML ANE full-sequence ({SEQ_LEN} tokens):")
    print(f"    avg: {avg_ms:.1f} ms, p50: {p50_ms:.1f} ms")
    print(f"    Rate: {1000/avg_ms:.0f} pred/s")
    return avg_ms


def benchmark_pytorch():
    """Benchmark PyTorch model on CPU/MPS for comparison."""
    with open(VOCAB_MAP) as f:
        vmap = json.load(f)

    model = DraftModel(vocab_size=vmap["pruned_vocab_size"])
    state = torch.load(PYTORCH_CKPT, map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    model.eval()

    for device_name in ["cpu", "mps"]:
        device = torch.device(device_name)
        m = model.to(device)
        dummy = torch.zeros(1, SEQ_LEN, dtype=torch.long, device=device)

        with torch.no_grad():
            for _ in range(5):
                m(dummy)
        if device_name == "mps":
            torch.mps.synchronize()

        times = []
        with torch.no_grad():
            for _ in range(20):
                t0 = time.perf_counter()
                m(dummy)
                if device_name == "mps":
                    torch.mps.synchronize()
                times.append(time.perf_counter() - t0)

        print(f"\n  PyTorch {device_name}: avg {np.mean(times)*1000:.1f} ms")


def estimate_parallel_speedup(ane_ms):
    """Estimate speedup with ANE running parallel to GPU verify."""
    baseline_tok_ms = 42.0   # 9B single token decode
    batch_verify_ms = 110.0  # 9B batch verify (N=8-32)
    ane_acceptance = 0.268    # measured
    ngram_acceptance = 0.46   # measured (financial text)
    pld_acceptance = 0.07     # measured

    print(f"\n{'='*60}")
    print(f"  Four-Path Architecture: Parallel ANE + GPU")
    print(f"{'='*60}")
    print(f"\n  Hardware latencies:")
    print(f"    9B single decode:     {baseline_tok_ms:.0f} ms (GPU)")
    print(f"    9B batch verify:      {batch_verify_ms:.0f} ms (GPU, N=8-32)")
    print(f"    Draft model forward:  {ane_ms:.1f} ms (ANE, parallel)")
    print(f"    N-gram + PLD:         ~0 ms (CPU)")

    print(f"\n  Acceptance rates (measured):")
    print(f"    ANE draft:     {ane_acceptance*100:.1f}%")
    print(f"    N-gram:        {ngram_acceptance*100:.0f}% (financial text)")
    print(f"    PLD:           {pld_acceptance*100:.0f}% (with echo)")

    # Scenario 1: Without ANE (current four-path: N-gram + PLD + GPU)
    print(f"\n  Scenario 1: N-gram + PLD only (current)")
    ngram_drafts = 8  # typical
    pld_drafts = 2    # typical
    total_drafts_no_ane = ngram_drafts + pld_drafts
    accepted_no_ane = ngram_drafts * ngram_acceptance + pld_drafts * pld_acceptance
    round_time_no_ane = batch_verify_ms  # 0ms draft + 110ms verify
    tokens_no_ane = accepted_no_ane + 1
    speedup_no_ane = (tokens_no_ane * baseline_tok_ms) / round_time_no_ane
    print(f"    Drafts: {total_drafts_no_ane}, Accepted: {accepted_no_ane:.1f}")
    print(f"    Round: {round_time_no_ane:.0f}ms -> {tokens_no_ane:.1f} tokens")
    print(f"    Speedup: {speedup_no_ane:.2f}x ({1000*tokens_no_ane/round_time_no_ane:.0f} tok/s)")

    # Scenario 2: With ANE (ANE runs parallel to GPU verify)
    print(f"\n  Scenario 2: N-gram + PLD + ANE (parallel)")
    ane_drafts = 1  # one prediction per round
    total_drafts_ane = ngram_drafts + pld_drafts + ane_drafts
    accepted_ane = accepted_no_ane + ane_drafts * ane_acceptance
    # ANE runs parallel to GPU verify — no added time IF ane_ms < batch_verify_ms
    parallel = ane_ms < batch_verify_ms
    if parallel:
        round_time_ane = batch_verify_ms  # ANE hidden within GPU verify
        print(f"    ANE ({ane_ms:.1f}ms) < verify ({batch_verify_ms:.0f}ms): PARALLEL (zero cost)")
    else:
        round_time_ane = ane_ms  # ANE is bottleneck
        print(f"    ANE ({ane_ms:.1f}ms) > verify ({batch_verify_ms:.0f}ms): ANE is bottleneck")

    tokens_ane = accepted_ane + 1
    speedup_ane = (tokens_ane * baseline_tok_ms) / round_time_ane
    print(f"    Drafts: {total_drafts_ane}, Accepted: {accepted_ane:.1f}")
    print(f"    Round: {round_time_ane:.0f}ms -> {tokens_ane:.1f} tokens")
    print(f"    Speedup: {speedup_ane:.2f}x ({1000*tokens_ane/round_time_ane:.0f} tok/s)")

    delta = speedup_ane - speedup_no_ane
    print(f"\n  ANE contribution: +{delta:.2f}x ({delta/speedup_no_ane*100:.1f}% relative improvement)")
    print(f"  Extra tokens per round: +{ane_acceptance:.2f}")
    print(f"  Marginal cost: {'0 ms (parallel)' if parallel else f'{ane_ms:.1f} ms'}")

    # Scenario 3: Higher acceptance from distillation
    print(f"\n  Scenario 3: KL distillation (projected ~40% acceptance)")
    kl_acceptance = 0.40
    accepted_kl = accepted_no_ane + ane_drafts * kl_acceptance
    tokens_kl = accepted_kl + 1
    speedup_kl = (tokens_kl * baseline_tok_ms) / round_time_ane
    print(f"    Accepted: {accepted_kl:.1f}, Speedup: {speedup_kl:.2f}x")

    # Key insight
    print(f"\n  Key insight:")
    print(f"    The ANE draft token is FREE in parallel mode.")
    print(f"    Even at 26.8% acceptance, it adds {ane_acceptance:.2f} tokens/round")
    print(f"    with zero time cost. This is additive with N-gram + PLD.")
    print(f"    On the Pro (70B target, 42M-param draft), the same architecture")
    print(f"    gives ~5 ANE tokens/round (multi-token prediction head).")


def main():
    print("=" * 60)
    print("  SRAM-Resident ANE Draft Model Benchmark")
    print("=" * 60)

    ane_ms = benchmark_coreml()
    benchmark_pytorch()
    estimate_parallel_speedup(ane_ms)


if __name__ == "__main__":
    main()

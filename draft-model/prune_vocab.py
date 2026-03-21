#!/usr/bin/env python3
"""
Build pruned 30K vocabulary from Qwen3.5 248K tokenizer.

Tokenizes the financial corpus (10-K, S-1 filings) + ISDA domain terms,
takes the top 30K most frequent tokens, builds a bidirectional mapping.

Output: vocab_map.json with:
  - pruned_to_full: pruned_id -> full Qwen3.5 token_id
  - full_to_pruned: full Qwen3.5 token_id -> pruned_id (or -1 for UNK)
  - pruned_vocab_size: 30000
"""

import json
import os
import sys
from collections import Counter
from pathlib import Path

# Must run from a directory without profile.py to avoid stdlib shadow
assert not os.path.exists("profile.py"), "Run from /tmp or another dir without profile.py"

from transformers import AutoTokenizer

CORPUS_DIRS = [
    "/Users/midas/Desktop/cowork/ngram-engine/10k-samples",
    "/Users/midas/Desktop/cowork/ngram-engine/s1-samples",
]

DOMAIN_TERMS = [
    "ISDA", "CSA", "collateral", "netting", "counterparty", "derivative",
    "swap", "margin", "threshold", "default", "termination", "close-out",
    "valuation", "exposure", "hedge", "bilateral", "cleared", "OTC",
    "notional", "mark-to-market", "MTM", "variation", "initial margin",
    "eligible collateral", "transfer", "delivery", "pledgor", "secured party",
    "non-defaulting", "determining party", "calculation agent", "commercially",
    "reasonable", "procedures", "outstanding", "transactions",
    "Early Termination Date", "Event of Default", "Credit Support Annex",
    "Close-out Amount", "Terminated Transaction", "Minimum Transfer Amount",
    "Independent Amount", "Valuation Date", "Valuation Agent",
    "Schedule", "Section", "Article", "Agreement", "Master Agreement",
    "Confirmation", "Transaction", "Party", "Obligations", "Payment",
    "interest rate", "credit default", "total return", "equity",
    "commodity", "foreign exchange", "cross-currency", "basis swap",
    "overnight", "SOFR", "LIBOR", "benchmark", "fallback",
    "Basel III", "Basel IV", "SA-CCR", "CVA", "XVA", "MVA", "FVA",
    "regulatory capital", "risk-weighted", "leverage ratio",
]

TARGET_VOCAB_SIZE = 30000
OUTPUT_DIR = "/Users/midas/Desktop/cowork/ane-perf/draft-model"


def main():
    print("Loading Qwen3.5 tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("mlx-community/Qwen3.5-9B-MLX-4bit")
    full_vocab_size = len(tokenizer)
    print(f"Full vocab: {full_vocab_size}")

    # Count token frequencies across corpus
    counter = Counter()
    total_tokens = 0

    for corpus_dir in CORPUS_DIRS:
        for f in sorted(Path(corpus_dir).glob("*.txt")):
            text = f.read_text()
            tokens = tokenizer.encode(text)
            counter.update(tokens)
            total_tokens += len(tokens)
            print(f"  {f.name}: {len(tokens):,} tokens")

    print(f"\nTotal corpus: {total_tokens:,} tokens, {len(counter):,} unique")

    # Ensure domain terms are included
    domain_token_ids = set()
    for term in DOMAIN_TERMS:
        ids = tokenizer.encode(term, add_special_tokens=False)
        domain_token_ids.update(ids)
    print(f"Domain term tokens: {len(domain_token_ids)}")

    # Always include special tokens
    special_ids = set()
    for attr in ["bos_token_id", "eos_token_id", "pad_token_id"]:
        tid = getattr(tokenizer, attr, None)
        if tid is not None:
            special_ids.add(tid)
    # Include common special tokens from Qwen
    for tid in range(min(200, full_vocab_size)):
        special_ids.add(tid)
    print(f"Special token IDs: {len(special_ids)}")

    # Build pruned vocab: specials + domain + top-frequency
    must_include = special_ids | domain_token_ids
    remaining_slots = TARGET_VOCAB_SIZE - len(must_include)

    # Sort by frequency, exclude must-include
    freq_sorted = [tid for tid, _ in counter.most_common()
                   if tid not in must_include]
    top_freq = set(freq_sorted[:remaining_slots])

    selected = must_include | top_freq
    # Pad to TARGET_VOCAB_SIZE with common tokens (low IDs are frequent in BPE)
    if len(selected) < TARGET_VOCAB_SIZE:
        for tid in range(full_vocab_size):
            if tid not in selected:
                selected.add(tid)
            if len(selected) >= TARGET_VOCAB_SIZE:
                break

    pruned_full_ids = sorted(selected)[:TARGET_VOCAB_SIZE]

    # Build mappings
    pruned_to_full = {i: fid for i, fid in enumerate(pruned_full_ids)}
    full_to_pruned = {fid: i for i, fid in enumerate(pruned_full_ids)}

    # Stats
    covered = sum(counter[tid] for tid in pruned_full_ids if tid in counter)
    coverage = covered / total_tokens * 100

    print(f"\nPruned vocab: {len(pruned_to_full)}")
    print(f"Corpus coverage: {coverage:.1f}% of tokens")

    # Check domain term coverage
    domain_covered = sum(1 for tid in domain_token_ids if tid in full_to_pruned)
    print(f"Domain terms covered: {domain_covered}/{len(domain_token_ids)}")

    # Save
    output = {
        "pruned_to_full": pruned_to_full,
        "full_to_pruned": full_to_pruned,
        "pruned_vocab_size": len(pruned_to_full),
        "full_vocab_size": full_vocab_size,
        "corpus_tokens": total_tokens,
        "coverage_pct": round(coverage, 2),
    }

    out_path = os.path.join(OUTPUT_DIR, "vocab_map.json")
    with open(out_path, "w") as f:
        json.dump(output, f)
    print(f"\nSaved to {out_path}")

    # Quick decode test
    test = "The ISDA Master Agreement provides for close-out netting"
    full_ids = tokenizer.encode(test, add_special_tokens=False)
    pruned_ids = [full_to_pruned.get(tid, 0) for tid in full_ids]
    recovered = [pruned_to_full[pid] for pid in pruned_ids]
    recovered_text = tokenizer.decode(recovered)
    print(f"\nRoundtrip test:")
    print(f"  Input:     {test}")
    print(f"  Recovered: {recovered_text}")
    print(f"  Match: {test == recovered_text}")


if __name__ == "__main__":
    main()

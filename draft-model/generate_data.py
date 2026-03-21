#!/usr/bin/env python3
"""
Generate training data for draft model from financial corpus.

Phase 1 (this script): Hard-label next-token prediction.
  - Tokenize corpus with Qwen3.5 tokenizer
  - Map to pruned 30K vocab
  - Save as training chunks

Phase 2 (later, if needed): KL distillation against 9B soft targets.

No teacher model needed for Phase 1 — just the corpus and tokenizer.
"""

import json
import os
import sys
import numpy as np
from pathlib import Path

assert os.path.basename(os.getcwd()) != "ane-perf", \
    "Run from /tmp to avoid profile.py shadowing stdlib"

from transformers import AutoTokenizer

CORPUS_DIRS = [
    "/Users/midas/Desktop/cowork/ngram-engine/10k-samples",
    "/Users/midas/Desktop/cowork/ngram-engine/s1-samples",
]
VOCAB_MAP_PATH = "/Users/midas/Desktop/cowork/ane-perf/draft-model/vocab_map.json"
OUTPUT_DIR = "/Users/midas/Desktop/cowork/ane-perf/draft-model/train_data"

CHUNK_SIZE = 512
STRIDE = 256


def main():
    # Load vocab map
    with open(VOCAB_MAP_PATH) as f:
        vmap = json.load(f)
    full_to_pruned = {int(k): v for k, v in vmap["full_to_pruned"].items()}
    print(f"Pruned vocab: {vmap['pruned_vocab_size']}")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained("mlx-community/Qwen3.5-9B-MLX-4bit")

    # Tokenize and chunk
    all_chunks = []
    for corpus_dir in CORPUS_DIRS:
        for f in sorted(Path(corpus_dir).glob("*.txt")):
            text = f.read_text()
            tokens = tokenizer.encode(text)
            # Map to pruned vocab
            pruned = [full_to_pruned.get(tid, 0) for tid in tokens]
            # Chunk with overlap
            n_chunks = 0
            for i in range(0, len(pruned) - CHUNK_SIZE, STRIDE):
                all_chunks.append(pruned[i:i + CHUNK_SIZE])
                n_chunks += 1
            print(f"  {f.name}: {len(tokens):,} tokens -> {n_chunks} chunks")

    print(f"\nTotal: {len(all_chunks)} chunks of {CHUNK_SIZE} tokens")

    # Split input/target
    data = np.array(all_chunks, dtype=np.int32)
    input_ids = data[:, :-1]   # (N, 511)
    target_ids = data[:, 1:]   # (N, 511)

    # Check UNK rate
    unk_count = np.sum(input_ids == 0)
    total = input_ids.size
    print(f"UNK tokens: {unk_count}/{total} ({100*unk_count/total:.2f}%)")

    # Save
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    np.save(os.path.join(OUTPUT_DIR, "input_ids.npy"), input_ids)
    np.save(os.path.join(OUTPUT_DIR, "target_ids.npy"), target_ids)

    print(f"\nSaved to {OUTPUT_DIR}/")
    print(f"  input_ids.npy:  {input_ids.shape}")
    print(f"  target_ids.npy: {target_ids.shape}")
    print(f"  Total token positions: {input_ids.shape[0] * input_ids.shape[1]:,}")

    # Quick sanity check: decode a sample
    sample_pruned = input_ids[100, :20].tolist()
    pruned_to_full = {int(k): v for k, v in vmap["pruned_to_full"].items()}
    sample_full = [pruned_to_full[pid] for pid in sample_pruned]
    print(f"\n  Sample text: {tokenizer.decode(sample_full)}")


if __name__ == "__main__":
    main()

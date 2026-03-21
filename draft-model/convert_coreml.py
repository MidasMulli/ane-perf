#!/usr/bin/env python3
"""
Convert trained draft model to CoreML INT8 for ANE deployment.

Creates a CoreML-friendly version of the model with fixed sequence length
and pre-computed buffers (no dynamic slicing or int ops).
"""

import os
import sys
import json
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model import DraftModel, RMSNorm

CHECKPOINT = "/Users/midas/Desktop/cowork/ane-perf/draft-model/checkpoints/best.pt"
VOCAB_MAP = "/Users/midas/Desktop/cowork/ane-perf/draft-model/vocab_map.json"
OUTPUT_PATH = "/Users/midas/Desktop/cowork/ane-perf/draft-model/draft_model.mlpackage"
SEQ_LEN = 512


def make_rotate_half(half_dim):
    """Create a rotate_half function with compile-time constant split."""
    def rotate_half(x):
        x1 = x[..., :half_dim]
        x2 = x[..., half_dim:]
        return torch.cat((-x2, x1), dim=-1)
    return rotate_half


class CoreMLAttention(nn.Module):
    """Attention with pre-computed RoPE and causal mask baked in.
    All dimensions are compile-time constants for CoreML tracing."""
    def __init__(self, dim, n_heads, seq_len):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.dim = dim
        self.seq_len = seq_len
        self.scale = 1.0 / math.sqrt(self.head_dim)
        self.qkv_proj = nn.Linear(dim, dim * 3, bias=False)
        self.o_proj = nn.Linear(dim, dim, bias=False)

        # Pre-compute RoPE
        inv_freq = 1.0 / (10000 ** (torch.arange(0, self.head_dim, 2).float() / self.head_dim))
        t = torch.arange(seq_len, dtype=inv_freq.dtype)
        freqs = torch.einsum("i,j->ij", t, inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos().unsqueeze(0).unsqueeze(0))
        self.register_buffer("sin_cached", emb.sin().unsqueeze(0).unsqueeze(0))

        # Pre-compute causal mask
        mask = torch.full((seq_len, seq_len), -1e4)
        mask = torch.triu(mask, diagonal=1)
        self.register_buffer("causal_mask", mask)
        self._rotate_half = make_rotate_half(self.head_dim // 2)

    def forward(self, x):
        qkv = self.qkv_proj(x)
        q = qkv[:, :, :self.dim]
        k = qkv[:, :, self.dim:2*self.dim]
        v = qkv[:, :, 2*self.dim:]
        q = q.reshape(1, self.seq_len, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        k = k.reshape(1, self.seq_len, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        v = v.reshape(1, self.seq_len, self.n_heads, self.head_dim).permute(0, 2, 1, 3)

        q = (q * self.cos_cached) + (self._rotate_half(q) * self.sin_cached)
        k = (k * self.cos_cached) + (self._rotate_half(k) * self.sin_cached)

        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attn = attn + self.causal_mask
        attn = F.softmax(attn, dim=-1)
        out = torch.matmul(attn, v)
        out = out.permute(0, 2, 1, 3).reshape(1, self.seq_len, self.dim)
        return self.o_proj(out)


class CoreMLFeedForward(nn.Module):
    def __init__(self, dim, ffn_dim):
        super().__init__()
        self.gate_proj = nn.Linear(dim, ffn_dim, bias=False)
        self.up_proj = nn.Linear(dim, ffn_dim, bias=False)
        self.down_proj = nn.Linear(ffn_dim, dim, bias=False)

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class CoreMLBlock(nn.Module):
    def __init__(self, dim, n_heads, ffn_dim, seq_len):
        super().__init__()
        self.attn_norm = RMSNorm(dim)
        self.attn = CoreMLAttention(dim, n_heads, seq_len)
        self.ffn_norm = RMSNorm(dim)
        self.ffn = CoreMLFeedForward(dim, ffn_dim)

    def forward(self, x):
        x = x + self.attn(self.attn_norm(x))
        x = x + self.ffn(self.ffn_norm(x))
        return x


class CoreMLDraftModel(nn.Module):
    """CoreML-friendly draft model with fixed seq_len and no dynamic ops."""
    def __init__(self, vocab_size=30000, dim=1024, n_layers=6,
                 n_heads=8, ffn_dim=2048, seq_len=512):
        super().__init__()
        self.embed_tokens = nn.Embedding(vocab_size, dim)
        self.layers = nn.ModuleList([
            CoreMLBlock(dim, n_heads, ffn_dim, seq_len) for _ in range(n_layers)
        ])
        self.norm = RMSNorm(dim)
        self.lm_head = nn.Linear(dim, vocab_size, bias=False)
        # Tie weights
        self.lm_head.weight = self.embed_tokens.weight

    def forward(self, input_ids):
        x = self.embed_tokens(input_ids)
        for layer in self.layers:
            x = layer(x)
        x = self.norm(x)
        return self.lm_head(x)


def transfer_weights(src_model, dst_model):
    """Transfer weights from trained DraftModel to CoreMLDraftModel."""
    src_sd = src_model.state_dict()
    dst_sd = dst_model.state_dict()

    transferred = 0
    for name, param in dst_sd.items():
        # Map CoreML buffer names
        src_name = name
        if src_name in src_sd and src_sd[src_name].shape == param.shape:
            dst_sd[name] = src_sd[src_name]
            transferred += 1

    # Handle the weight mappings between different structures
    for i in range(6):
        prefix_src = f"layers.{i}."
        prefix_dst = f"layers.{i}."
        for suffix in ["attn_norm.weight", "ffn_norm.weight",
                       "attn.qkv_proj.weight", "attn.o_proj.weight",
                       "ffn.gate_proj.weight", "ffn.up_proj.weight",
                       "ffn.down_proj.weight"]:
            src_key = prefix_src + suffix
            dst_key = prefix_dst + suffix
            if src_key in src_sd and dst_key in dst_sd:
                dst_sd[dst_key] = src_sd[src_key]

    # Embedding and norm
    dst_sd["embed_tokens.weight"] = src_sd["embed_tokens.weight"]
    dst_sd["norm.weight"] = src_sd["norm.weight"]
    # lm_head is tied to embed_tokens

    dst_model.load_state_dict(dst_sd, strict=False)
    return dst_model


def main():
    import coremltools as ct

    with open(VOCAB_MAP) as f:
        vmap = json.load(f)
    vocab_size = vmap["pruned_vocab_size"]

    # Load trained model
    print("Loading trained model...")
    src_model = DraftModel(vocab_size=vocab_size)
    state = torch.load(CHECKPOINT, map_location="cpu", weights_only=True)
    src_model.load_state_dict(state)
    src_model.eval()

    # Create CoreML-friendly model and transfer weights
    print("Creating CoreML-friendly model...")
    coreml_model = CoreMLDraftModel(vocab_size=vocab_size, seq_len=SEQ_LEN)
    coreml_model = transfer_weights(src_model, coreml_model)
    coreml_model.eval()

    # Verify outputs match
    print("Verifying weight transfer...")
    test_input = torch.randint(0, vocab_size, (1, SEQ_LEN))
    with torch.no_grad():
        out_src = src_model(test_input)
        out_dst = coreml_model(test_input)
    diff = (out_src - out_dst).abs().max().item()
    print(f"  Max output diff: {diff:.6f}")
    assert diff < 0.01, f"Output mismatch too large: {diff}"

    # Trace
    print(f"Tracing with seq_len={SEQ_LEN}...")
    traced = torch.jit.trace(coreml_model, test_input)

    # Convert to CoreML
    print("Converting to CoreML...")
    mlmodel = ct.convert(
        traced,
        inputs=[ct.TensorType(name="input_ids", shape=(1, SEQ_LEN),
                              dtype=np.int32)],
        outputs=[ct.TensorType(name="logits")],
        convert_to="mlprogram",
        minimum_deployment_target=ct.target.macOS15,
    )

    # Quantize to INT8
    print("Quantizing to INT8...")
    mlmodel_q = ct.compression_utils.affine_quantize_weights(
        mlmodel, mode="linear", dtype=np.int8
    )

    # Save
    print(f"Saving to {OUTPUT_PATH}...")
    mlmodel_q.save(OUTPUT_PATH)

    size_mb = sum(
        os.path.getsize(os.path.join(dp, f))
        for dp, dn, fn in os.walk(OUTPUT_PATH)
        for f in fn
    ) / 1e6
    print(f"Model size: {size_mb:.1f} MB")

    print(f"\nVerify ANE placement:")
    print(f"  python3 /Users/midas/Desktop/cowork/ane-perf/profile.py {OUTPUT_PATH}")
    print(f"  python3 /Users/midas/Desktop/cowork/ane-perf/measure.py {OUTPUT_PATH} --idle -n 200")


if __name__ == "__main__":
    main()

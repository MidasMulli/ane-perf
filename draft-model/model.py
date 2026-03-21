#!/usr/bin/env python3
"""
SRAM-resident ANE draft model for speculative decoding against Qwen3.5-9B.

Architecture constraints (from IOReport measurements):
  - Per-layer weights < 19 MB (SRAM-resident zone)
  - Total model weights > 37 MB (CoreML ANE dispatch threshold)
  - INT8 weights (1.58-1.90x faster than FP16 on ANE)
  - Same tokenizer as Qwen3.5-9B (248K vocab, required for spec decode)

Model: 6-layer transformer decoder
  dim=1024, FFN=2048, 8 heads, head_dim=128
  Per-layer INT8: ~10.4 MB (well under 19 MB SRAM limit)
  6 layers = 62.4 MB (above 37 MB ANE dispatch threshold)
  Vocab: pruned to 30K most common Qwen3.5 tokens
  Embedding + lm_head: 30K * 1024 * 1 byte = ~30 MB
  Total: ~93 MB
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        norm = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return x * norm * self.weight


class RotaryEmbedding(nn.Module):
    def __init__(self, dim, max_seq_len=2048):
        super().__init__()
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)
        self.max_seq_len = max_seq_len

    def forward(self, seq_len, device):
        t = torch.arange(seq_len, device=device, dtype=self.inv_freq.dtype)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        return emb.cos(), emb.sin()


def rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin):
    cos = cos.unsqueeze(0).unsqueeze(0)  # (1, 1, seq, dim)
    sin = sin.unsqueeze(0).unsqueeze(0)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class Attention(nn.Module):
    def __init__(self, dim, n_heads):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.qkv_proj = nn.Linear(dim, dim * 3, bias=False)
        self.o_proj = nn.Linear(dim, dim, bias=False)

    def forward(self, x, cos, sin, mask=None):
        B, T, C = x.shape
        qkv = self.qkv_proj(x)
        q, k, v = qkv.chunk(3, dim=-1)
        q = q.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        q, k = apply_rotary_pos_emb(q, k, cos[:T], sin[:T])
        attn = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        if mask is not None:
            attn = attn + mask[:T, :T]
        attn = F.softmax(attn, dim=-1)
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        return self.o_proj(out)


class FeedForward(nn.Module):
    def __init__(self, dim, ffn_dim):
        super().__init__()
        self.gate_proj = nn.Linear(dim, ffn_dim, bias=False)
        self.up_proj = nn.Linear(dim, ffn_dim, bias=False)
        self.down_proj = nn.Linear(ffn_dim, dim, bias=False)

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class TransformerBlock(nn.Module):
    def __init__(self, dim, n_heads, ffn_dim):
        super().__init__()
        self.attn_norm = RMSNorm(dim)
        self.attn = Attention(dim, n_heads)
        self.ffn_norm = RMSNorm(dim)
        self.ffn = FeedForward(dim, ffn_dim)

    def forward(self, x, cos, sin, mask=None):
        x = x + self.attn(self.attn_norm(x), cos, sin, mask)
        x = x + self.ffn(self.ffn_norm(x))
        return x


class DraftModel(nn.Module):
    """
    SRAM-resident draft model for ANE speculative decoding.

    Args:
        vocab_size: pruned vocabulary size (maps to/from full Qwen3.5 248K vocab)
        dim: model dimension
        n_layers: number of transformer layers
        n_heads: number of attention heads
        ffn_dim: feed-forward intermediate dimension
        max_seq_len: maximum sequence length
    """
    def __init__(self, vocab_size=30000, dim=1024, n_layers=6,
                 n_heads=8, ffn_dim=2048, max_seq_len=2048):
        super().__init__()
        self.vocab_size = vocab_size
        self.dim = dim
        self.n_layers = n_layers

        self.embed_tokens = nn.Embedding(vocab_size, dim)
        self.layers = nn.ModuleList([
            TransformerBlock(dim, n_heads, ffn_dim) for _ in range(n_layers)
        ])
        self.norm = RMSNorm(dim)
        self.lm_head = nn.Linear(dim, vocab_size, bias=False)

        # Tie weights
        self.lm_head.weight = self.embed_tokens.weight

        self.rotary = RotaryEmbedding(dim // n_heads, max_seq_len)

        # Causal mask
        mask = torch.full((max_seq_len, max_seq_len), float("-inf"))
        mask = torch.triu(mask, diagonal=1)
        self.register_buffer("causal_mask", mask)

    def forward(self, input_ids):
        B, T = input_ids.shape
        x = self.embed_tokens(input_ids)
        cos, sin = self.rotary(T, x.device)
        for layer in self.layers:
            x = layer(x, cos, sin, self.causal_mask)
        x = self.norm(x)
        logits = self.lm_head(x)
        return logits

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters())

    def layer_sizes_int8(self):
        """Compute per-layer INT8 weight sizes in bytes."""
        sizes = {}
        for i, layer in enumerate(self.layers):
            total = 0
            for name, p in layer.named_parameters():
                if "weight" in name and p.dim() >= 2:
                    total += p.numel()  # 1 byte per param in INT8
            sizes[f"layer_{i}"] = total
        embed_size = self.embed_tokens.weight.numel()
        sizes["embedding"] = embed_size
        sizes["lm_head"] = embed_size  # tied
        return sizes


def print_model_specs():
    """Print model specifications and verify SRAM constraints."""
    model = DraftModel()
    params = model.count_parameters()
    sizes = model.layer_sizes_int8()

    print("=" * 60)
    print("  SRAM-Resident ANE Draft Model Specifications")
    print("=" * 60)
    print(f"\n  Parameters: {params:,} ({params/1e6:.1f}M)")
    print(f"  Vocab: {model.vocab_size:,} (pruned from 248K)")
    print(f"  Dim: {model.dim}, Layers: {model.n_layers}")
    print(f"\n  Per-layer INT8 sizes:")

    total = 0
    for name, size in sizes.items():
        mb = size / 1e6
        total += size
        sram_ok = "SRAM" if mb < 19 else "DRAM"
        print(f"    {name:15s}: {mb:6.1f} MB  [{sram_ok}]")

    print(f"\n  Total INT8: {total/1e6:.1f} MB")
    print(f"  ANE dispatch threshold (37 MB): {'PASS' if total/1e6 > 37 else 'FAIL'}")

    # Verify constraints
    layer_sizes = [v for k, v in sizes.items() if k.startswith("layer_")]
    max_layer = max(layer_sizes) / 1e6
    print(f"  Max layer size: {max_layer:.1f} MB (limit: 19 MB): "
          f"{'PASS' if max_layer < 19 else 'FAIL'}")


if __name__ == "__main__":
    print_model_specs()

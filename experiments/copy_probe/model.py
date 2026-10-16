"""Tiny byte-level causal transformer used as a reference for the copy probes.

The ``bug`` field injects one known defect so the probes in ``probes.py`` can
be tested against models that are wrong in a specific way. Real runs use
``bug=None``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

BOS, EOS, PAD = 256, 257, 258
VOCAB_SIZE = 259
IGNORE = -100

# noncausal:  attention can see future positions (label leak).
# window2:    each query sees only itself and the two previous positions, so
#             answer positions cannot read identifiers far back in the prompt.
# detach_kv:  keys and values are detached. The forward pass is unchanged but
#             no gradient reaches the prompt, so retrieval cannot be learned.
BUGS = (None, "noncausal", "window2", "detach_kv")


@dataclass(frozen=True)
class ModelConfig:
    d_model: int = 128
    n_layers: int = 2
    n_heads: int = 4
    max_len: int = 160
    bug: str | None = None

    def __post_init__(self) -> None:
        if self.bug not in BUGS:
            raise ValueError(f"unknown bug {self.bug!r}")
        if self.d_model % self.n_heads:
            raise ValueError("d_model must be divisible by n_heads")


def attention_mask(length: int, bug: str | None, device=None) -> torch.Tensor:
    """Boolean [T, T] mask; True means the query row may attend to the key column."""
    i = torch.arange(length, device=device)[:, None]
    j = torch.arange(length, device=device)[None, :]
    if bug == "noncausal":
        return torch.ones(length, length, dtype=torch.bool, device=device)
    mask = j <= i
    if bug == "window2":
        mask &= i - j <= 2
    return mask


class Attention(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.n_heads = cfg.n_heads
        self.bug = cfg.bug
        self.qkv = nn.Linear(cfg.d_model, 3 * cfg.d_model)
        self.out = nn.Linear(cfg.d_model, cfg.d_model)

    def forward(self, x: torch.Tensor, mask: torch.Tensor):
        b, t, d = x.shape
        qkv = self.qkv(x).view(b, t, 3, self.n_heads, d // self.n_heads)
        q, k, v = qkv.permute(2, 0, 3, 1, 4)
        if self.bug == "detach_kv":
            k, v = k.detach(), v.detach()
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(q.shape[-1])
        weights = scores.masked_fill(~mask, float("-inf")).softmax(-1)
        y = (weights @ v).transpose(1, 2).reshape(b, t, d)
        return self.out(y), weights


class Block(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.d_model)
        self.attn = Attention(cfg)
        self.ln2 = nn.LayerNorm(cfg.d_model)
        self.mlp = nn.Sequential(
            nn.Linear(cfg.d_model, 4 * cfg.d_model),
            nn.GELU(),
            nn.Linear(4 * cfg.d_model, cfg.d_model),
        )

    def forward(self, x: torch.Tensor, mask: torch.Tensor):
        a, weights = self.attn(self.ln1(x), mask)
        x = x + a
        return x + self.mlp(self.ln2(x)), weights


class TinyByteLM(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(VOCAB_SIZE, cfg.d_model)
        self.pos_emb = nn.Embedding(cfg.max_len, cfg.d_model)
        self.blocks = nn.ModuleList(Block(cfg) for _ in range(cfg.n_layers))
        self.ln_f = nn.LayerNorm(cfg.d_model)
        self.head = nn.Linear(cfg.d_model, VOCAB_SIZE, bias=False)
        self.apply(self._init)

    @staticmethod
    def _init(module: nn.Module) -> None:
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, std=0.02)
        if isinstance(module, nn.Linear) and module.bias is not None:
            nn.init.zeros_(module.bias)

    def forward(self, ids: torch.Tensor, return_attention: bool = False):
        t = ids.shape[1]
        if t > self.cfg.max_len:
            raise ValueError(f"sequence length {t} exceeds max_len {self.cfg.max_len}")
        positions = torch.arange(t, device=ids.device)
        x = self.tok_emb(ids) + self.pos_emb(positions)
        mask = attention_mask(t, self.cfg.bug, ids.device)
        attention = []
        for block in self.blocks:
            x, weights = block(x, mask)
            attention.append(weights)
        logits = self.head(self.ln_f(x))
        return (logits, attention) if return_attention else logits


def answer_loss(logits: torch.Tensor, labels: torch.Tensor, weights: torch.Tensor | None = None):
    """Mean cross-entropy over supervised positions (labels != IGNORE)."""
    flat = F.cross_entropy(
        logits.reshape(-1, logits.shape[-1]),
        labels.reshape(-1),
        ignore_index=IGNORE,
        reduction="none",
    ).view_as(labels)
    supervised = (labels != IGNORE).float()
    if weights is not None:
        supervised = supervised * weights
    return (flat * supervised).sum() / supervised.sum().clamp_min(1.0)


@torch.no_grad()
def greedy_generate(model: nn.Module, prompts: list[list[int]], max_new: int) -> list[list[int]]:
    """Greedy decoding without a KV cache.

    Prompts are grouped by length so no padding enters the attention window;
    each step re-runs the full forward pass. Returns generated ids up to and
    including EOS when it is produced.
    """
    model.eval()
    results: list[list[int] | None] = [None] * len(prompts)
    by_length: dict[int, list[int]] = {}
    for index, prompt in enumerate(prompts):
        by_length.setdefault(len(prompt), []).append(index)
    device = next(model.parameters()).device
    for indices in by_length.values():
        ids = torch.tensor([prompts[i] for i in indices], device=device)
        done = torch.zeros(len(indices), dtype=torch.bool, device=device)
        out: list[list[int]] = [[] for _ in indices]
        for _ in range(max_new):
            nxt = model(ids)[:, -1].argmax(-1)
            for row, token in enumerate(nxt.tolist()):
                if not done[row]:
                    out[row].append(token)
            done |= nxt == EOS
            if bool(done.all()):
                break
            ids = torch.cat([ids, nxt[:, None]], dim=1)
        for row, index in enumerate(indices):
            results[index] = out[row]
    return results  # type: ignore[return-value]

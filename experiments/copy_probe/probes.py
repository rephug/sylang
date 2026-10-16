"""Model-agnostic checks for whether a causal LM pipeline *can* learn to copy.

Each probe needs only callables, so it can be pointed at another training
codebase with a few lines of adapter code:

* ``logits_fn(ids: LongTensor[B, T]) -> FloatTensor[B, T, V]``, using the same
  forward path as training (same masks, positions and padding policy)
* ``loss_fn(logits, batch) -> scalar``, the exact training loss
* ``generate_fn(prompt: str) -> list[int]``, the exact evaluation decoder
* ``encode_fn(prompt: str) -> list[int]``, the training-time prompt encoding

Mechanical probes (1-7) can run on an *untrained* model; they test whether
information and gradients can flow from the prompt to the answer. The
behavioural probes (8-9) are for trained checkpoints.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable

import torch
from torch import nn


@dataclass
class ProbeResult:
    name: str
    passed: bool
    detail: dict = field(default_factory=dict)

    def __str__(self) -> str:
        status = "PASS" if self.passed else "FAIL"
        return f"[{status}] {self.name}: {self.detail}"


def _perturb(ids: torch.Tensor, position: int, vocab_size: int) -> torch.Tensor:
    changed = ids.clone()
    changed[:, position] = (changed[:, position] + 1 + vocab_size // 2) % vocab_size
    return changed


# 1 ------------------------------------------------------------------------
@torch.no_grad()
def causality_probe(logits_fn: Callable, ids: torch.Tensor, vocab_size: int, atol: float = 1e-6) -> ProbeResult:
    """Changing token j must not change logits at positions < j (no future leak)."""
    base = logits_fn(ids)
    worst = 0.0
    for j in range(1, ids.shape[1]):
        out = logits_fn(_perturb(ids, j, vocab_size))
        worst = max(worst, (out[:, :j] - base[:, :j]).abs().max().item())
    return ProbeResult("causality", worst <= atol, {"max_change_before_edit": worst})


# 2 ------------------------------------------------------------------------
@torch.no_grad()
def visibility_probe(logits_fn: Callable, ids: torch.Tensor, query: int, vocab_size: int, atol: float = 0.0) -> ProbeResult:
    """Changing any earlier token must change the logits at ``query``.

    Works at random initialisation: an exactly unchanged logit means that
    position is masked out or disconnected, not merely unlearned.
    """
    base = logits_fn(ids)[:, query]
    blind: list[int] = []
    changes: list[float] = []
    for j in range(query + 1):
        delta = (logits_fn(_perturb(ids, j, vocab_size))[:, query] - base).abs().max().item()
        changes.append(delta)
        if delta <= atol:
            blind.append(j)
    return ProbeResult(
        "visibility",
        not blind,
        {"query": query, "blind_positions": blind, "min_change": min(changes)},
    )


# 3 ------------------------------------------------------------------------
def gradient_reach_probe(
    model: nn.Module,
    embedding: nn.Module,
    batch: dict,
    loss_fn: Callable,
    logits_fn: Callable | None = None,
    supervised_key: str = "labels",
    ignore_index: int = -100,
    atol: float = 0.0,
    pad_id: int | None = None,
) -> ProbeResult:
    """The answer loss must send gradient to every prompt-token embedding.

    Forward probes cannot see stop-gradient defects (``detach``, ``no_grad``
    blocks, frozen key/value paths). This one can. Pass ``pad_id`` when pads
    precede the answer (left padding); masked pads may get zero gradient.
    """
    captured: list[torch.Tensor] = []

    def hook(_module, _inputs, output):
        output.retain_grad()
        captured.append(output)

    handle = embedding.register_forward_hook(hook)
    try:
        model.zero_grad(set_to_none=True)
        logits = (logits_fn or model)(batch["input_ids"])
        loss_fn(logits, batch).backward()
    finally:
        handle.remove()
    grad = captured[0].grad
    if grad is None:
        return ProbeResult("gradient_reach", False, {"error": "no gradient reached the embedding output"})
    norms = grad.norm(dim=-1)
    unreached = 0
    total = 0
    worst_row = None
    for row, labels in enumerate(batch[supervised_key]):
        supervised = (labels != ignore_index).nonzero().flatten()
        first = int(supervised.min())  # positions before this are pure prompt
        prompt = torch.arange(first)
        if pad_id is not None:
            prompt = prompt[batch["input_ids"][row, :first] != pad_id]
        dead_positions = prompt[norms[row, prompt] <= atol]
        unreached += len(dead_positions)
        total += len(prompt)
        if len(dead_positions) and worst_row is None:
            worst_row = {"row": row, "dead_positions": dead_positions.tolist()}
    model.zero_grad(set_to_none=True)
    return ProbeResult(
        "gradient_reach",
        unreached == 0,
        {"prompt_positions": total, "zero_gradient_positions": unreached, "example": worst_row},
    )


# 4 ------------------------------------------------------------------------
def parameter_gradient_probe(model: nn.Module, loss: torch.Tensor, optimizer=None) -> ProbeResult:
    """Every trainable tensor gets a nonzero gradient and is in the optimizer."""
    model.zero_grad(set_to_none=True)
    loss.backward()
    missing = [n for n, p in model.named_parameters() if p.requires_grad and (p.grad is None or not p.grad.any())]
    not_optimized: list[str] = []
    if optimizer is not None:
        in_optimizer = {id(p) for group in optimizer.param_groups for p in group["params"]}
        not_optimized = [n for n, p in model.named_parameters() if p.requires_grad and id(p) not in in_optimizer]
    frozen = [n for n, p in model.named_parameters() if not p.requires_grad]
    model.zero_grad(set_to_none=True)
    return ProbeResult(
        "parameter_gradients",
        not missing and not not_optimized,
        {"zero_or_none_grad": missing, "not_in_optimizer": not_optimized, "frozen": frozen},
    )


# 5 ------------------------------------------------------------------------
@torch.no_grad()
def oracle_label_probe(
    loss_fn: Callable,
    batch: dict,
    vocab_size: int,
    pad_id: int | None = None,
    sequences: list[list[int]] | None = None,
    offsets: list[int] | None = None,
) -> ProbeResult:
    """Checks the collate + loss pairing without a model.

    ``sequences`` are the full token sequences (BOS ... EOS) behind each row,
    starting at ``offsets[row]`` in ``input_ids`` (default 0). Pass them when
    the inputs omit the final target token (``x = seq[:-1]``); by default they
    are read from ``input_ids`` with padding stripped.

    * ``next``: logits[t] put all mass on the true next token. The training loss must be ~0.
    * ``same``: logits[t] put all mass on input[t]. The loss must be large;
      ~0 means labels are not shifted and the model is taught to echo.
    * ``wrong_final_token``: correct everywhere except the prediction of the
      final token of each sequence (normally EOS). The loss must rise; no rise
      means EOS is not supervised, so the model never learns to stop.
    """
    ids = batch["input_ids"]
    b, t = ids.shape
    if sequences is None:
        real = ids != pad_id if pad_id is not None else torch.ones_like(ids, dtype=torch.bool)
        sequences, offsets = [], []
        for row in range(b):
            positions = real[row].nonzero().flatten()
            first, last = int(positions.min()), int(positions.max())
            sequences.append(ids[row, first : last + 1].tolist())
            offsets.append(first)
    offsets = offsets or [0] * b

    def onehot(targets: torch.Tensor) -> torch.Tensor:
        logits = torch.full((b, t, vocab_size), -30.0)
        logits.scatter_(2, targets.unsqueeze(-1), 30.0)
        return logits

    next_ids = ids.clone()
    wrong_last = ids.clone()
    for row, (sequence, offset) in enumerate(zip(sequences, offsets)):
        for i in range(len(sequence) - 1):
            if offset + i < t:
                next_ids[row, offset + i] = sequence[i + 1]
        wrong_last[row] = next_ids[row]
        last = offset + len(sequence) - 2  # predicts the final token
        wrong_last[row, last] = (sequence[-1] + 1) % vocab_size
    loss_next = float(loss_fn(onehot(next_ids), batch))
    loss_same = float(loss_fn(onehot(ids), batch))
    loss_last = float(loss_fn(onehot(wrong_last), batch))
    passed = loss_next < 1e-3 and loss_same > 1.0 and loss_last > 1e-2
    return ProbeResult(
        "oracle_labels",
        passed,
        {"loss_next_token_oracle": loss_next, "loss_echo_oracle": loss_same, "loss_wrong_final_token": loss_last},
    )


# 6 ------------------------------------------------------------------------
@torch.no_grad()
def checkpoint_roundtrip_probe(
    model: nn.Module,
    build_fresh: Callable[[], nn.Module],
    save: Callable[[nn.Module], object],
    load: Callable[[nn.Module, object], None],
    ids: torch.Tensor,
) -> ProbeResult:
    """Logits after save -> load into a fresh model must equal the originals,
    and loading must actually change the fresh model."""
    model.eval()
    original = model(ids)
    handle = save(model)
    fresh = build_fresh()
    fresh.eval()
    before = fresh(ids)
    load(fresh, handle)
    after = fresh(ids)
    diff = (after - original).abs().max().item()
    moved = (before - original).abs().max().item()
    return ProbeResult(
        "checkpoint_roundtrip",
        diff == 0.0 and moved > 0.0,
        {"max_logit_diff_after_load": diff, "fresh_vs_trained_diff": moved},
    )


# 7 ------------------------------------------------------------------------
@torch.no_grad()
def decoder_consistency_probe(
    generate_fn: Callable[[str], list[int]],
    logits_fn: Callable,
    encode_fn: Callable[[str], list[int]],
    prompts: list[str],
) -> ProbeResult:
    """Each generated token must equal the argmax of a teacher-forced forward
    pass over the *training* encoding of the prompt plus the generated prefix.

    Catches evaluation-only differences: prompt formatting, BOS handling,
    position offsets, KV-cache errors, sampling settings, wrong stop logic.
    """
    mismatches = []
    for prompt in prompts:
        generated = generate_fn(prompt)
        context = encode_fn(prompt)
        logits = logits_fn(torch.tensor([context + generated]))[0]
        for i, token in enumerate(generated):
            predicted = int(logits[len(context) - 1 + i].argmax())
            if predicted != token:
                mismatches.append({"prompt": prompt[-40:], "step": i, "generated": token, "teacher_forced": predicted})
                break
    return ProbeResult("decoder_consistency", not mismatches, {"checked": len(prompts), "mismatches": mismatches[:3]})


# 7b -----------------------------------------------------------------------
@torch.no_grad()
def padding_invariance_probe(
    collate_fn: Callable[[list], dict],
    batch_logits_fn: Callable[[dict], torch.Tensor],
    examples: list,
    supervised_key: str = "labels",
    ignore_index: int = -100,
    atol: float = 1e-4,
) -> ProbeResult:
    """The logits of ``examples[0]`` at its supervised positions must not
    change when it is batched with the other, longer examples.

    Catches padding-side, attention-mask and position-offset defects that only
    appear in batched training with variable-length prompts.
    """
    alone = collate_fn(examples[:1])
    mixed = collate_fn(examples)

    def supervised_logits(batch):
        logits = batch_logits_fn(batch)[0]
        positions = (batch[supervised_key][0] != ignore_index).nonzero().flatten()
        return logits[positions]

    a, b = supervised_logits(alone), supervised_logits(mixed)
    diff = float((a - b).abs().max()) if a.shape == b.shape else math.inf
    return ProbeResult(
        "padding_invariance",
        diff <= atol,
        {"max_logit_diff": diff, "padded_width": int(mixed["input_ids"].shape[1]), "alone_width": int(alone["input_ids"].shape[1])},
    )


# 8 ------------------------------------------------------------------------
def _entropy_bits(p: torch.Tensor) -> torch.Tensor:
    return -(p * p.clamp_min(1e-30).log2()).sum(-1)


@torch.no_grad()
def prompt_dependence_probe(logits_fn: Callable, prompts: list[list[int]], threshold_bits: float = 0.05) -> ProbeResult:
    """How much does the next-token distribution at the answer position depend
    on the prompt?

    Reports the generalized Jensen-Shannon divergence between the per-prompt
    distributions (entropy of the mean minus mean entropy, in bits). Near zero
    means the model emits the same distribution for every prompt: it predicts
    a marginal, not a copy. A copying model scores up to log2(#distinct targets).
    """
    dists = []
    for prompt in prompts:
        dists.append(logits_fn(torch.tensor([prompt]))[0, -1].softmax(-1))
    p = torch.stack(dists)
    mean = p.mean(0)
    js = float(_entropy_bits(mean) - _entropy_bits(p).mean())
    top = mean.topk(3)
    return ProbeResult(
        "prompt_dependence",
        js > threshold_bits,
        {"js_bits": js, "mean_top_tokens": top.indices.tolist(), "mean_top_probs": [round(float(v), 4) for v in top.values]},
    )


# 9 ------------------------------------------------------------------------
@torch.no_grad()
def counterfactual_probe(logits_fn: Callable, pairs: list[tuple[list[int], list[int], int, int]]) -> dict:
    """For prompt pairs ``(a, b, target_a, target_b)`` that differ in one
    controlled way (e.g. swapped roles or renamed identifiers), report the
    total-variation distance between answer-position distributions and how
    much probability follows the correct target.
    """
    tv, follow = [], []
    for a, b, target_a, target_b in pairs:
        pa = logits_fn(torch.tensor([a]))[0, -1].softmax(-1)
        pb = logits_fn(torch.tensor([b]))[0, -1].softmax(-1)
        tv.append(0.5 * float((pa - pb).abs().sum()))
        if target_a != target_b:
            follow.append(float(pb[target_b] - pa[target_b]))
    return {
        "mean_tv": sum(tv) / len(tv),
        "mean_gain_on_new_target": sum(follow) / len(follow) if follow else math.nan,
        "pairs": len(pairs),
    }

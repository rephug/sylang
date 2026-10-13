"""Synthetic byte-level tasks for testing whether a model copies identifiers.

Every task yields ``Example(prompt, answer)``. The model sees
``[BOS] + prompt bytes + answer bytes + [EOS]`` and is supervised only on the
answer bytes and EOS.

Identifiers are letter-digit-letter strings such as ``a0r``. ``FreshIds``
draws each character independently, so the ending of an identifier cannot be
predicted from its first character; held-out identifiers are unseen
*combinations* of characters that all occur in training.
"""

from __future__ import annotations

import hashlib
import random
import string
from dataclasses import dataclass, field

import torch

from model import BOS, EOS, IGNORE, PAD

LETTERS = string.ascii_lowercase
DIGITS = string.digits
OTHER_RELATIONS = ("likes", "joins", "meets")  # same length as "helps"


@dataclass(frozen=True)
class Example:
    prompt: str
    answer: str
    kind: str  # "select", "none", "copy" or "repeat"
    targets: tuple[str, ...] = ()
    meta: dict = field(default_factory=dict, compare=False, hash=False)


# --------------------------------------------------------------------------
# Identifier sources


def all_identifiers() -> list[str]:
    return [a + d + b for a in LETTERS for d in DIGITS for b in LETTERS]


def identifier_split(identifier: str) -> str:
    """Stable 80/20 split by hash: 'train' or 'test'."""
    digest = hashlib.sha256(identifier.encode()).digest()
    return "test" if digest[0] % 5 == 0 else "train"


class FreshIds:
    """Uniform identifiers from one side of the stable split."""

    def __init__(self, split: str) -> None:
        self.ids = [i for i in all_identifiers() if identifier_split(i) == split]

    def sample(self, rng: random.Random, k: int) -> list[str]:
        return rng.sample(self.ids, k)


class PoolIds:
    """A fixed identifier pool, optionally with sampling weights."""

    def __init__(self, ids: list[str], weights: list[float] | None = None) -> None:
        self.ids = list(ids)
        self.weights = list(weights) if weights else [1.0] * len(ids)

    def sample(self, rng: random.Random, k: int) -> list[str]:
        chosen: list[str] = []
        while len(chosen) < k:
            candidate = rng.choices(self.ids, self.weights)[0]
            if candidate not in chosen:
                chosen.append(candidate)
        return chosen


def fixed_pool(split: str, size: int, seed: int) -> PoolIds:
    """``size`` uniform identifiers from a split.

    Caution: drawn from 6,760 combinations, 12 identifiers leave most held-out
    identifiers with a character never seen at its position in training, so
    held-out failure is guaranteed whether or not the model copies. Use
    ``balanced_pool`` for a small-pool design.
    """
    return PoolIds(random.Random(seed).sample(FreshIds(split).ids, size))


def balanced_pool(split: str, curriculum: str | None = None) -> PoolIds:
    """The project's balanced 12/12 COPY identifiers (alphabet a-f; every letter
    exactly twice at every position in each split), read at run time from the
    packet's ``copy/source/CURRICULUM.json``. The path comes from ``curriculum``
    or the ``SYLANG_COPY_CURRICULUM`` environment variable; the lists are not
    stored in this repository.
    """
    import json
    import os

    path = curriculum or os.environ.get("SYLANG_COPY_CURRICULUM")
    if not path:
        raise ValueError("balanced12 needs the packet's CURRICULUM.json (--curriculum or SYLANG_COPY_CURRICULUM)")
    with open(path, encoding="utf-8") as handle:
        data = json.load(handle)
    key = {"train": "train_identifiers", "test": "heldout_identifiers"}[split]
    return PoolIds(list(data[key]))


# --------------------------------------------------------------------------
# Task generators


def repeat_example(rng: random.Random, length: int = 6) -> Example:
    """Positive control: copy a random string after a separator."""
    text = "".join(rng.choice(LETTERS + DIGITS) for _ in range(length))
    return Example(prompt=f"{text}|", answer=text, kind="repeat", targets=(text,))


def _filler(rng: random.Random, jitter: int) -> str:
    return "~" * rng.randint(0, jitter) if jitter else ""


def copy_none_example(rng: random.Random, ids, jitter: int = 0, copy: bool | None = None) -> Example:
    """Single identifier; copy it when the mode is COPY, answer empty for NONE.

    With ``jitter=0`` every prompt has the same length. With ``jitter>0`` a
    filler of 0..jitter bytes moves the identifier, with the same length
    distribution for COPY and NONE.
    """
    copy = rng.random() < 0.5 if copy is None else copy
    (identifier,) = ids.sample(rng, 1)
    mode = "COPY" if copy else "NONE"
    prompt = f"{_filler(rng, jitter)}Mode: {mode}. Id: {identifier}.\nAnswer: "
    if copy:
        return Example(prompt, identifier, "copy", (identifier,))
    return Example(prompt, "", "none", (identifier,))


def role_select_example(
    rng: random.Random,
    ids,
    jitter: int = 0,
    select: bool | None = None,
    target_slot: int | None = None,
) -> Example:
    """Two facts with four distinct identifiers; answer ``subject/object`` of
    the fact whose relation is ``helps``, or the empty answer if none is.

    Selected and empty prompts have identical length for the same filler.
    The target fact's slot varies, so a fixed offset from the answer cannot
    solve it; COPY/NONE can be solved that way even with jitter.
    """
    select = rng.random() < 0.5 if select is None else select
    a, b, c, d = ids.sample(rng, 4)
    relations = list(rng.sample(OTHER_RELATIONS, 2))
    slot = None
    if select:
        slot = rng.randrange(2) if target_slot is None else target_slot
        relations[slot] = "helps"
    prompt = (
        f"{_filler(rng, jitter)}Facts: {a} {relations[0]} {b}. {c} {relations[1]} {d}. "
        "Who helps whom?\nAnswer: "
    )
    meta = {"pairs": ((a, b), (c, d)), "slot": slot, "relations": tuple(relations)}
    if not select:
        return Example(prompt, "", "none", (), meta)
    subject, obj = (a, b) if slot == 0 else (c, d)
    return Example(prompt, f"{subject}/{obj}", "select", (subject, obj), meta)


def swap_roles(example: Example) -> Example:
    """Swap subject and object of the target fact (role probe)."""
    if example.kind != "select":
        raise ValueError("only select examples have roles")
    subject, obj = example.targets
    marker = f"{subject} helps {obj}."
    if marker not in example.prompt:
        raise ValueError("target fact not found")
    prompt = example.prompt.replace(marker, f"{obj} helps {subject}.")
    return Example(prompt, f"{obj}/{subject}", "select", (obj, subject), example.meta)


def replace_identifiers(example: Example, mapping: dict[str, str]) -> Example:
    """Rename identifiers everywhere (content probe)."""
    prompt, answer = example.prompt, example.answer
    for old, new in mapping.items():
        if len(old) != len(new):
            raise ValueError("replacement must keep byte length")
        prompt = prompt.replace(old, "\0" + new)
        answer = answer.replace(old, "\0" + new)
    prompt, answer = prompt.replace("\0", ""), answer.replace("\0", "")
    targets = tuple(mapping.get(t, t) for t in example.targets)
    return Example(prompt, answer, example.kind, targets, example.meta)


# --------------------------------------------------------------------------
# Replica of the reported failing regime


REPLICA_IDS = (
    "a0r", "g4k", "b3x", "c5m", "d1q", "e8v",
    "f2j", "h6z", "i9p", "j7w", "k3n", "l5y",
)  # every ending is fixed by the first character


def replica_pool() -> PoolIds:
    """Zipf-weighted pool, so ``a0r`` is the most frequent target."""
    return PoolIds(list(REPLICA_IDS), [1.0 / (rank + 1) for rank in range(len(REPLICA_IDS))])


def replica_example(
    rng: random.Random, pool: PoolIds, select: bool, pad_none: bool = False, distractor: str = "likes"
) -> Example:
    """English prompt whose length reveals the class unless ``pad_none``.

    Selected prompts hold a distractor fact and a target fact; empty prompts
    hold only the distractor, so they are shorter. ``pad_none`` adds a second
    distractor to an empty prompt, giving it exactly the selected length with
    only trained words (probe for the length shortcut). With
    ``distractor="sees"`` (the first variant run) the padded prompt is one
    byte shorter than a selected prompt.
    """
    s, o, x, y = pool.sample(rng, 4)
    facts = [f"{x} {distractor} {y}."]
    if select:
        facts.append(f"{s} helps {o}.")
    elif pad_none:
        facts.append(f"{s} {distractor} {o}.")
    rng.shuffle(facts)
    prompt = f"Facts: {' '.join(facts)} Who helps whom?\nAnswer: "
    if select:
        return Example(prompt, f"{s}/{o}", "select", (s, o))
    return Example(prompt, "", "none")


def replica_datasets(seed: int, n_train: int = 64, n_eval_each: int = 24, n_seen: int = 8, distractor: str = "likes"):
    """64 training examples; 24 empty and 24 selected checks, 8 selected seen.

    Returns ``(train, checks, length_matched_none)``.
    """
    rng = random.Random(seed)
    pool = replica_pool()
    train: list[Example] = []
    while len(train) < n_train:
        ex = replica_example(rng, pool, select=len(train) % 2 == 0, distractor=distractor)
        if ex not in train:
            train.append(ex)
    seen = [ex for ex in train if ex.kind == "select"][:n_seen]
    unseen: list[Example] = []
    while len(unseen) < n_eval_each - n_seen:
        ex = replica_example(rng, pool, select=True, distractor=distractor)
        if ex not in train and ex not in unseen:
            unseen.append(ex)
    empty: list[Example] = []
    while len(empty) < n_eval_each:
        ex = replica_example(rng, pool, select=False, distractor=distractor)
        if ex not in train and ex not in empty:
            empty.append(ex)
    padded_none = [
        replica_example(rng, pool, select=False, pad_none=True, distractor=distractor) for _ in range(n_eval_each)
    ]
    for ex in seen:
        ex.meta["seen"] = True
    return train, seen + unseen + empty, padded_none


# --------------------------------------------------------------------------
# Encoding


def prompt_ids(prompt: str) -> list[int]:
    return [BOS, *prompt.encode()]


def encode(example: Example) -> tuple[list[int], int]:
    """Return ``(ids, answer_start)``; ``ids[answer_start]`` is the first answer byte."""
    ids = prompt_ids(example.prompt)
    start = len(ids)
    return ids + list(example.answer.encode()) + [EOS], start


def answer_roles(example: Example) -> list[str]:
    """Role of each supervised target: id_first, id_rest, sep or eos."""
    roles: list[str] = []
    at_start = True
    for char in example.answer:
        if char == "/":
            roles.append("sep")
            at_start = True
        else:
            roles.append("id_first" if at_start else "id_rest")
            at_start = False
    return roles + ["eos"]


def collate(examples: list[Example], first_byte_weight: float = 1.0, bug: str | None = None) -> dict:
    """Right-padded batch; ``labels[t]`` is the target predicted from position ``t``.

    Probe-test defects: ``bug="unshifted"`` is an off-by-one label defect;
    ``bug="left_pad"`` pads on the left without masking or position offsets,
    so a sequence's logits depend on the other sequences in its batch.
    """
    encoded = [encode(ex) for ex in examples]
    width = max(len(ids) for ids, _ in encoded)
    input_ids = torch.full((len(examples), width), PAD, dtype=torch.long)
    labels = torch.full((len(examples), width), IGNORE, dtype=torch.long)
    weights = torch.zeros((len(examples), width))
    for row, (example, (ids, start)) in enumerate(zip(examples, encoded)):
        shift = width - len(ids) if bug == "left_pad" else 0
        input_ids[row, shift : shift + len(ids)] = torch.tensor(ids)
        for offset, role in enumerate(answer_roles(example)):
            source = start - 1 + offset
            t = shift + source
            labels[row, t] = ids[source] if bug == "unshifted" else ids[source + 1]
            weights[row, t] = first_byte_weight if role == "id_first" else 1.0
    return {"input_ids": input_ids, "labels": labels, "weights": weights}

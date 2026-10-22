#!/usr/bin/env python3
"""Recompute the new checks in docs/research/diffusion-review-2026-10-22.md.

Finite-world arithmetic only: the repository's codec, the pinned Qwen3
tokenizer and the unmodified FiniteOracle. No model is trained or run. Round
counts are idealized sampler counts, not inference steps or latency.
"""

from __future__ import annotations

import argparse
from collections import Counter
import itertools
import json
import math
from pathlib import Path
import statistics
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from sylang_core import encode  # noqa: E402 (repository modules need ROOT on the path)
from sylang_core.codec import (
    ASPECTS, EVIDENCES, POLARITIES, RELATIONS, SCHEMA_VERSION, TENSES, _CODES,
)
from evaluation.diffusion_experiment import FIELDS, PAD, build_worlds, semantic_record
from evaluation.diffusion_oracle import FiniteOracle
from evaluation.diffusion_surfaces import _tokenizer, tokenize_surface
from evaluation.harness import load_suite

TOKENIZER = "qwen3"
ENGLISH_TAG = "Sylang core-v0.1: "
M_TAG = "M0.1:"
CONCISE_EVIDENCE = {
    "Without specified evidence, ": "",
    "By direct observation, ": "Directly, ",
    "By report, ": "Reportedly, ",
    "By inference, ": "Presumably, ",
}
LABEL_PAIRS = (("A", "B"), ("Priya", "Fatima"), ("Priya", "Xiomara"))


def tokens(text: str) -> tuple:
    return tokenize_surface(text, TOKENIZER)


def concise_english(ast: dict) -> str:
    text = encode(ast, "english")[len(ENGLISH_TAG):]
    for phrase, replacement in CONCISE_EVIDENCE.items():
        if text.startswith(phrase):
            text = replacement + text[len(phrase):]
            break
    for label in {ast["statement"]["subject"], ast["statement"]["object"]}:
        text = text.replace(json.dumps(label, ensure_ascii=False), label)
    return text


def tuple_json(ast: dict, codes: bool) -> str:
    node = ast["statement"]
    values = [node[f] if f in ("subject", "object") or not codes else _CODES[f][node[f]]
              for f in FIELDS]
    return json.dumps(values, ensure_ascii=False, separators=(",", ":"))


VARIANTS = {
    "english_codec": lambda ast: encode(ast, "english"),
    "english_body_without_version_tag": lambda ast: encode(ast, "english")[len(ENGLISH_TAG):],
    "english_concise": concise_english,
    "json_named_codec": lambda ast: encode(ast, "json"),
    "json_tuple_words": lambda ast: tuple_json(ast, codes=False),
    "json_tuple_m_codes": lambda ast: tuple_json(ast, codes=True),
    "m_codec": lambda ast: encode(ast, "m"),
    "m_body_without_version_tag": lambda ast: encode(ast, "m")[len(M_TAG):],
}


def header_split() -> dict:
    """Payload tokens with and without only the mandatory per-record version tag."""
    world = build_worlds()["independent_864"]
    rows, _ = load_suite()
    counter = _tokenizer(TOKENIZER, None)
    result = {}
    for name, documents, count in (
        ("independent_864", world, lambda text: len(tokens(text))),
        # The harness counts the two transport-diagnostic fixtures too, so use its rule here.
        ("smoke_suite_33", [row["ast"] for row in rows], lambda text: counter.measure(text)["tokens"]),
    ):
        totals = {}
        for fmt, tag in (("english", ENGLISH_TAG), ("m", M_TAG)):
            texts = [encode(doc, fmt) for doc in documents]
            if not all(text.startswith(tag) for text in texts):
                raise AssertionError(f"Unexpected {fmt} version tag")
            totals[fmt] = {"with_tag": sum(map(count, texts)),
                           "without_tag": sum(count(text[len(tag):]) for text in texts)}
        result[name] = {
            "records": len(documents),
            "tokens": totals,
            "m_over_english_with_tag": totals["m"]["with_tag"] / totals["english"]["with_tag"],
            "m_over_english_without_tag": totals["m"]["without_tag"] / totals["english"]["without_tag"],
        }
    return result


def baseline_lengths() -> dict:
    world = build_worlds()["independent_864"]
    result = {}
    for name, render in VARIANTS.items():
        texts = [render(ast) for ast in world]
        if len(set(texts)) != len(world):
            raise AssertionError(f"{name} is not injective on the 864-meaning world")
        lengths = [len(tokens(text)) for text in texts]
        result[name] = {"mean_tokens": statistics.fmean(lengths),
                        "min_tokens": min(lengths), "max_tokens": max(lengths)}
    return result


def labelled_world(labels: tuple[str, str]) -> list[dict]:
    return [{"version": SCHEMA_VERSION, "statement": {"kind": "pred", **dict(zip(FIELDS, values))}}
            for values in itertools.product(labels, RELATIONS, labels, POLARITIES,
                                            TENSES, ASPECTS, EVIDENCES)]


def fixed_canvas_oracle(documents: list[dict], fmt: str) -> FiniteOracle:
    streams = [tokens(encode(doc, fmt)) for doc in documents]
    width = max(map(len, streams))
    return FiniteOracle([s + (PAD,) * (width - len(s)) for s in streams],
                        [semantic_record(doc) for doc in documents])


def adaptive_rounds(oracle: FiniteOracle) -> dict:
    result = {}
    for order in ("forward", "reverse"):
        score = oracle.independent_group_refinement(oracle.full_mask, order)
        result[order] = {"mean": score["mean_number_rounds"], "worst": score["worst_number_rounds"]}
    one = oracle.completion_stats(oracle.full_mask, [list(range(len(oracle.streams[0])))])
    result["one_round_valid_support"] = one["valid_support_mass"]
    return result


def label_sensitivity() -> dict:
    result = {}
    for labels in LABEL_PAIRS:
        documents = labelled_world(labels)
        result["/".join(labels)] = {
            fmt: {"mean_tokens": statistics.fmean(len(tokens(encode(d, fmt))) for d in documents),
                  **adaptive_rounds(fixed_canvas_oracle(documents, fmt))}
            for fmt in ("english", "json", "m")}
    return result


def prior_perturbation() -> dict:
    """Drop one meaning: total correlation becomes tiny, but exact factorization fails."""
    documents = build_worlds()["independent_864"][:-1]
    records = [semantic_record(doc) for doc in documents]
    entropy = math.log2(len(records))
    marginals = math.fsum(
        -count / len(records) * math.log2(count / len(records))
        for field in range(len(FIELDS))
        for count in Counter(r[field] for r in records).values())
    return {"meanings": len(documents), "total_correlation_bits": marginals - entropy,
            "m": adaptive_rounds(fixed_canvas_oracle(documents, "m"))}


def threshold_rounds(oracle: FiniteOracle, tau: float) -> dict:
    """Exact-marginal analogue of confidence-threshold parallel decoding.

    Each round commits every undecided position whose exact marginal confidence
    is at least tau, plus the single most confident uncertain position if none
    qualifies; committed positions are sampled independently. Branches are
    enumerated exactly; invalid joint draws are absorbing failures.
    """
    width = len(oracle.streams[0])
    states = [(oracle.full_mask, frozenset(), 0.0, 0)]
    finished: dict[int, float] = {}
    while states:
        following = []
        for candidates, done, log_probability, rounds in states:
            if len(done) == width:
                finished[rounds] = finished.get(rounds, 0.0) + math.exp(log_probability)
                continue
            count = candidates.bit_count()
            marginals = {}
            for position in range(width):
                if position not in done:
                    marginals[position] = {
                        token: hits for token, mask in oracle.token_masks[position].items()
                        if (hits := (candidates & mask).bit_count())}
            confidence = {p: max(c.values()) / count for p, c in marginals.items()}
            group = [p for p in sorted(confidence) if confidence[p] >= tau]
            if all(confidence[p] == 1.0 for p in group):
                uncertain = [p for p in sorted(confidence) if confidence[p] < 1.0]
                if uncertain:
                    group.append(max(uncertain, key=lambda p: (confidence[p], -p)))
            branches: dict[tuple, int] = {}
            for index in range(oracle.size):
                if candidates >> index & 1:
                    key = tuple(oracle.streams[index][p] for p in group)
                    branches[key] = branches.get(key, 0) | (1 << index)
            for key, branch in branches.items():
                step = math.fsum(math.log(marginals[p][t] / count) for p, t in zip(group, key))
                following.append((branch, done | frozenset(group), log_probability + step, rounds + 1))
        states = following
    valid = math.fsum(finished.values())
    return {"tau": tau, "valid_mass": valid,
            "mean_rounds_given_valid": math.fsum(r * m for r, m in finished.items()) / valid,
            "worst_rounds": max(finished)}


def threshold_sampler(tau: float = 0.9) -> dict:
    worlds = build_worlds()
    return {world: {fmt: threshold_rounds(fixed_canvas_oracle(worlds[world], fmt), tau)
                    for fmt in ("english", "json", "m")}
            for world in ("independent_864", "correlated_18")}


def substitutions() -> dict:
    """Replace one token by another token seen at that position in an equal-length stream."""
    world = build_worlds()["independent_864"]
    result = {}
    for fmt in ("english", "json", "m"):
        streams = [tokens(encode(doc, fmt)) for doc in world]
        valid = set(streams)
        alternatives = {}
        for stream in streams:
            slots = alternatives.setdefault(len(stream), [set() for _ in stream])
            for position, token in enumerate(stream):
                slots[position].add(token)
        silent = detectable = 0
        for stream in streams:
            for position, token in enumerate(stream):
                for other in alternatives[len(stream)][position] - {token}:
                    changed = stream[:position] + (other,) + stream[position + 1:]
                    if changed in valid:
                        silent += 1
                    else:
                        detectable += 1
        result[fmt] = {"substitutions": silent + detectable, "silent_valid_other_meaning": silent,
                       "detectable_invalid": detectable,
                       "detectable_fraction": detectable / (silent + detectable)}
    return result


def run() -> dict:
    return {
        "schema": "sylang-diffusion-review-checks-v1",
        "tokenizer": TOKENIZER,
        "methodology": {
            "model_training": False, "neural_inference": False,
            "rounds": "Exact expectations for named idealized samplers on the fixed public canvas; not inference steps or latency.",
            "version_tag": f"Only the codec's mandatory prefixes {ENGLISH_TAG!r} and {M_TAG!r} are removed; bodies are retokenized.",
            "concise_english": "Labels unquoted, version tag removed, unspecified evidence unmarked, other evidence as Directly/Reportedly/Presumably. Injective on this world only; not a codec.",
        },
        "version_tag_split": header_split(),
        "baseline_lengths_independent_864": baseline_lengths(),
        "label_sensitivity_fixed_canvas": label_sensitivity(),
        "drop_one_meaning_fixed_canvas": prior_perturbation(),
        "confidence_threshold_sampler_fixed_canvas": threshold_sampler(),
        "single_token_substitutions_independent_864": substitutions(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, help="Write deterministic JSON to this path")
    args = parser.parse_args()
    report = run()
    text = json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
        print(f"Wrote {args.output}; no model training or inference.")
    else:
        print(text, end="")


if __name__ == "__main__":
    main()

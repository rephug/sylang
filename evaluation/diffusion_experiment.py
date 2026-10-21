"""Finite-world CPU experiments; exact posteriors, no neural model execution.

Surface masking expectations are stratified Monte Carlo over masks. Semantic
masking, length information, and named completion schedules are enumerated.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
from pathlib import Path
import platform
import random
import statistics

from sylang_core import decode, encode
from sylang_core.codec import (
    ASPECTS, EVIDENCES, POLARITIES, RELATIONS, SCHEMA_VERSION, TENSES,
)
from .diffusion_fusion import run_fusion_experiments
from .diffusion_oracle import FiniteOracle
from .diffusion_surfaces import (
    decode_surface_tokens, surface_tokens, tokenizer_metadata,
)

ROOT = Path(__file__).resolve().parents[1]
FIELDS = ("subject", "relation", "object", "polarity", "tense", "aspect", "evidence")
LABELS = ("A", "B")
RATES = (0.0, 0.25, 0.5, 0.75, 0.9, 1.0)
METRICS = ("entropy_bits", "bayes_recovery", "unique_recovery",
           "conditional_total_correlation_bits")
PAD = ("fixed-canvas-padding",)
NATIVE_GROUPS = {
    "factorized_7": ((0,), (1,), (2,), (3,), (4,), (5,), (6,)),
    "fused_72_states_4": ((0,), (1,), (2,), (3, 4, 5, 6)),
    "fused_216_states_3": ((0,), (2,), (1, 3, 4, 5, 6)),
}


def semantic_record(ast: dict) -> tuple:
    return tuple(ast["statement"][field] for field in FIELDS)


def build_worlds() -> dict[str, list[dict]]:
    """Complete independent universe and an explicitly artificial correlated prior."""
    independent = []
    by_record = {}
    for values in itertools.product(LABELS, RELATIONS, LABELS, POLARITIES,
                                    TENSES, ASPECTS, EVIDENCES):
        ast = {"version": SCHEMA_VERSION,
               "statement": {"kind": "pred", **dict(zip(FIELDS, values))}}
        independent.append(ast)
        by_record[values] = ast
    correlated = []
    for subject, relation, tense in itertools.product(range(2), range(3), range(3)):
        values = (LABELS[subject], RELATIONS[relation], LABELS[1 - subject],
                  POLARITIES[(subject + relation) % 2], TENSES[tense],
                  ASPECTS[tense], EVIDENCES[(subject + relation + tense) % 4])
        correlated.append(by_record[values])
    return {"independent_864": independent, "correlated_18": correlated}


def average_metrics(weighted_stats) -> dict:
    totals = dict.fromkeys(METRICS, 0.0)
    total_weight = 0.0
    for weight, stats in weighted_stats:
        total_weight += weight
        for key in METRICS:
            totals[key] += weight * stats[key]
    if not math.isclose(total_weight, 1.0, abs_tol=1e-9):
        raise ValueError(f"Observation weights sum to {total_weight}, not one")
    return totals


def _check_rates(rates) -> None:
    if not rates or len(set(rates)) != len(rates) or any(
        type(p) not in (int, float) or not math.isfinite(p) or not 0 <= p <= 1 for p in rates
    ):
        raise ValueError("Mask probabilities must be unique finite numbers in [0, 1]")


def exact_unit_masking(oracle: FiniteOracle, rates=RATES) -> list[dict]:
    """Enumerate all masks and observation classes on constant-size native slots."""
    _check_rates(rates)
    lengths = {len(stream) for stream in oracle.streams}
    if len(lengths) != 1:
        raise ValueError("Semantic-unit enumeration requires constant-size streams")
    n = lengths.pop()
    partitions = []
    for hidden in itertools.product((False, True), repeat=n):
        visible = [index for index, erased in enumerate(hidden) if not erased]
        groups = {}
        for index, stream in enumerate(oracle.streams):
            observation = tuple(stream[position] for position in visible)
            groups[observation] = groups.get(observation, 0) | (1 << index)
        partitions.append((sum(hidden), list(groups.values())))
    output = []
    for probability in rates:
        observations = []
        for erased, groups in partitions:
            mask_weight = probability ** erased * (1 - probability) ** (n - erased)
            if not mask_weight:
                continue
            for candidates in groups:
                observations.append((mask_weight * candidates.bit_count() / oracle.size,
                                     oracle.posterior_stats(candidates)))
        output.append({"mask_probability": probability, "expectation": "exact_enumeration",
                       **average_metrics(observations)})
    return output


def surface_masking(oracle: FiniteOracle, samples_per_state: int, seed: int,
                    rates=RATES) -> list[dict]:
    """Enumerate worlds, sample masks; compute the exact posterior for every mask.

    Common random uniforms couple the mask-rate comparisons within each arm.
    SE estimates refer only to sampled masks, not general-language uncertainty.
    """
    _check_rates(rates)
    if type(samples_per_state) is not int or samples_per_state < 2:
        raise ValueError("At least two masks per meaning are required")
    sums = {p: dict.fromkeys(METRICS, 0.0) for p in rates}
    variances = {p: dict.fromkeys(METRICS, 0.0) for p in rates}
    for state, stream in enumerate(oracle.streams):
        rng = random.Random(seed + 104729 * state)
        uniforms = [tuple(rng.random() for _ in stream) for _ in range(samples_per_state)]
        for probability in rates:
            if probability == 0:
                samples = [oracle.posterior_stats(1 << state)]
            elif probability == 1:
                samples = [oracle.posterior_stats(oracle.length_masks[len(stream)])]
            else:
                samples = [oracle.posterior_stats(oracle.posterior(
                    stream, tuple(value < probability for value in draw))) for draw in uniforms]
            for key in METRICS:
                values = [sample[key] for sample in samples]
                sums[probability][key] += statistics.fmean(values) / oracle.size
                if len(values) > 1:
                    variances[probability][key] += (
                        statistics.variance(values) / len(values) / oracle.size ** 2)
    return [{"mask_probability": p,
             "expectation": "exact_enumeration" if p in (0, 1) else "stratified_mask_monte_carlo",
             "meaning_count": oracle.size,
             "masks_per_meaning": 1 if p in (0, 1) else samples_per_state,
             **sums[p],
             "mask_standard_error": {key: math.sqrt(value) for key, value in variances[p].items()}}
            for p in rates]


def length_information(oracle: FiniteOracle) -> dict:
    metrics = average_metrics((mask.bit_count() / oracle.size, oracle.posterior_stats(mask))
                              for mask in oracle.length_masks.values())
    prior_entropy = math.log2(oracle.size)
    return {"prior_entropy_bits": prior_entropy,
            "length_only_information_bits": max(0.0, prior_entropy - metrics["entropy_bits"]),
            "length_only": metrics,
            "unknown_length_full_erasure": oracle.posterior_stats(oracle.full_mask)}


def round_groups(length: int, budget: int, order: str) -> list[list[int]]:
    """Static left-to-right or interleaved groups; all positions get one prediction."""
    if type(length) is not int or type(budget) is not int or length <= 0 or budget <= 0 or order not in ("contiguous", "interleaved"):
        raise ValueError("Expected positive sizes and a declared order")
    groups = min(length, budget)
    if order == "contiguous":
        return [list(range(index * length // groups, (index + 1) * length // groups))
                for index in range(groups)]
    return [list(range(index, length, groups)) for index in range(groups)]


def refinement_schedules(oracle: FiniteOracle, threshold=0.99, *,
                         include_static=True, condition="full_erasure_gold_canvas_length_given") -> dict:
    """Exactly score named conditional samplers under full erasure + given length.

    Within a round positions are sampled independently from pre-round marginals.
    Commitments are sampled predictions, never hidden gold tokens. Probabilities
    of every valid final string are evaluated analytically by target factoring.
    """
    buckets = []
    aggregate = {name: [] for name in ("contiguous", "interleaved")} if include_static else {}
    adaptive_aggregate = {name: [] for name in ("forward", "reverse")}
    for length, candidates in sorted(oracle.length_masks.items()):
        budgets = sorted({min(length, 2 ** power) for power in range(9)} | {length})
        schedules = {}
        for order in aggregate:
            scores = [oracle.completion_stats(candidates, round_groups(length, budget, order))
                      for budget in budgets]
            successful = [score for score in scores
                          if score["valid_support_mass"] >= threshold - 1e-12
                          and score["total_variation_to_posterior"] <= 1 - threshold + 1e-12]
            first = min(successful, key=lambda score: score["number_rounds"]) if successful else None
            schedules[order] = {"tested": scores,
                                "minimum_tested_rounds_meeting_threshold":
                                first["number_rounds"] if first else None}
            if first is None:
                raise AssertionError("Singleton conditional sampling must reproduce posterior")
            aggregate[order].append((candidates.bit_count() / oracle.size, length, first))
        adaptive = {order: oracle.independent_group_refinement(candidates, order=order)
                    for order in adaptive_aggregate}
        for order, score in adaptive.items():
            adaptive_aggregate[order].append((candidates.bit_count() / oracle.size, score))
        buckets.append({"canvas_positions": length, "meaning_count": candidates.bit_count(),
                        "prior_weight": candidates.bit_count() / oracle.size,
                        "bayes_original_recovery_ceiling": 1 / candidates.bit_count(),
                        "exact_joint_sampler_rounds": 1,
                        "one_round_independent_marginals": oracle.completion_stats(candidates, [list(range(length))]),
                        "adaptive_independent_groups": adaptive, "schedules": schedules})
    summary = {}
    for order, weighted in aggregate.items():
        summary[order] = {
            "mean_minimum_tested_rounds": sum(w * score["number_rounds"] for w, n, score in weighted),
            "worst_minimum_tested_rounds": max(score["number_rounds"] for w, n, score in weighted),
            "mean_full_canvas_linear_work_proxy": sum(w * score["number_rounds"] * n for w, n, score in weighted),
            "mean_full_canvas_quadratic_work_proxy": sum(w * score["number_rounds"] * n ** 2 for w, n, score in weighted),
        }
    adaptive_summary = {}
    for order, weighted in adaptive_aggregate.items():
        adaptive_summary[order] = {
            "mean_number_rounds": sum(weight * score["mean_number_rounds"] for weight, score in weighted),
            "worst_number_rounds": max(score["worst_number_rounds"] for weight, score in weighted),
            "mean_full_canvas_linear_work_proxy": sum(weight * score["mean_processed_positions"] for weight, score in weighted),
            "mean_full_canvas_quadratic_work_proxy": sum(weight * score["mean_quadratic_work_proxy"] for weight, score in weighted),
        }
    return {"condition": condition,
            "valid_support_threshold": threshold, "tv_threshold": 1 - threshold,
            "selection": "smallest tested budget satisfying both thresholds in each length bucket",
            "summary": summary, "adaptive_summary": adaptive_summary, "length_buckets": buckets}


def run_experiments(tokenizers=("lexical", "utf8_bytes"), samples_per_state=8,
                    seed=20261020, progress=None) -> dict:
    if type(samples_per_state) is not int or samples_per_state < 2:
        raise ValueError("At least two masks per meaning are needed for mask standard errors")
    worlds = build_worlds()
    results = {}
    tokenizer_info = {name: tokenizer_metadata(name) for name in tokenizers}
    for name, documents in worlds.items():
        records = [semantic_record(document) for document in documents]
        native = {}
        for encoding, groups in NATIVE_GROUPS.items():
            streams = [tuple(tuple((FIELDS[index], record[index]) for index in group)
                             for group in groups) for record in records]
            oracle = FiniteOracle(streams, records)
            native[encoding] = {
                "unit_kind": "hypothetical_native_semantic_slot_not_existing_model_token",
                "slots_per_record": len(groups),
                "fields_per_slot": [[FIELDS[index] for index in group] for group in groups],
                "observed_vocabulary_size": len({value for stream in streams for value in stream}),
                "masking": exact_unit_masking(oracle),
                "refinement": refinement_schedules(oracle),
            }
        arms = {}
        for tokenizer, format_name in itertools.product(tokenizers, ("english", "json", "m")):
            arm_name = f"{tokenizer}/{format_name}"
            if progress:
                progress(f"{name}: {arm_name}")
            streams = [surface_tokens(document, format_name, tokenizer) for document in documents]
            for document, stream in zip(documents, streams):
                text = decode_surface_tokens(stream, tokenizer)
                if text != encode(document, format_name) or decode(text, format_name) != document:
                    raise AssertionError("Finite world transport or codec round trip failed")
            oracle = FiniteOracle(streams, records)
            lengths = [len(stream) for stream in streams]
            arm_seed = seed + int.from_bytes(hashlib.sha256(arm_name.encode()).digest()[:4], "big")
            padded = [stream + (PAD,) * (max(lengths) - len(stream)) for stream in streams]
            padded_oracle = FiniteOracle(padded, records)
            arms[arm_name] = {
                "format": format_name, "tokenizer": tokenizer,
                "density": {"mean_positions": statistics.fmean(lengths),
                            "min_positions": min(lengths), "max_positions": max(lengths),
                            "mean_utf8_bytes": statistics.fmean(len(encode(doc, format_name).encode()) for doc in documents),
                            "observed_token_vocabulary": len({value for stream in streams for value in stream}),
                            "semantic_entropy_per_mean_position": math.log2(len(records)) / statistics.fmean(lengths)},
                "length_information": length_information(oracle),
                "token_masking_gold_length_given": surface_masking(oracle, samples_per_state, arm_seed),
                "token_masking_fixed_canvas": {
                    "canvas_positions": max(lengths),
                    "padding_masked_identically_to_content": True,
                    "masking": surface_masking(padded_oracle, samples_per_state, arm_seed)},
                "refinement": refinement_schedules(oracle),
                "refinement_fixed_canvas": refinement_schedules(
                    padded_oracle, include_static=False, condition="full_erasure_fixed_public_canvas"),
            }
        corpus_bytes = json.dumps(documents, sort_keys=True, ensure_ascii=False,
                                  separators=(",", ":")).encode()
        results[name] = {
            "meaning_count": len(documents), "prior": "uniform over this world's declared support",
            "entropy_bits": math.log2(len(documents)),
            "canonical_corpus_sha256": hashlib.sha256(corpus_bytes).hexdigest(),
            "matched_semantic_field_masking": {
                "observation": "revealed field paths and values only; no surface text, length, or padding",
                "formats": ["english", "json", "m"],
                "representation_invariant_by_construction": True,
                "masking": native["factorized_7"]["masking"]},
            "native_semantic_ablations": native, "surface_arms": arms,
        }
    source_files = ("evaluation/diffusion_experiment.py", "evaluation/diffusion_oracle.py",
                    "evaluation/diffusion_fusion.py", "evaluation/diffusion_surfaces.py",
                    "sylang_core/codec.py", "benchmarks/tokenizers.lock.json")
    return {
        "schema": "sylang-cpu-diffusion-oracle-v1",
        "parameters": {"seed": seed, "samples_per_state": samples_per_state,
                       "mask_probabilities": list(RATES), "tokenizers": list(tokenizers)},
        "environment": {"python": platform.python_version(), "dependencies": "stdlib plus optional pinned tokenizers only"},
        "source_sha256": {path: hashlib.sha256((ROOT / path).read_bytes()).hexdigest() for path in source_files},
        "tokenizer_metadata": tokenizer_info,
        "methodology": {
            "model_training": False, "neural_inference": False,
            "topology": "public fixed single-predicate core-v0.1; no conditionals or length prediction",
            "fields": list(FIELDS), "literal_values": list(LABELS),
            "independent_world": "Complete Cartesian product: 2*3*2*2*3*3*4=864 meanings, uniform",
            "correlated_world": "18 uniform latent states (subject_index s,relation_index r,tense_index t): object=1-s; polarity=(s+r)%2; aspect=t; evidence=(s+r+t)%4",
            "posterior": "Exact finite-support posterior for every observation. Bernoulli token mask likelihood is constant for matching equal-length candidates.",
            "mask_expectations": "Surface interiors stratify over all meanings and sample masks; endpoints, semantic masks, length information and completion distributions are exact. Mask SE excludes corpus/prior uncertainty.",
            "length_leakage": "Gold-length conditions reveal target token count. Fixed-canvas controls pad to the public maximum; visible padding still conveys information, but full erasure cannot reveal target length.",
            "completion": "Round groups are static schedules, not optimal neural denoising. Each round independently samples exact marginals given prior sampled commitments. Invalid prefixes fail. All target probabilities computed by enumeration, without revealing source gold.",
            "adaptive_completion": "Greedy exact joint-independence checks choose parallel groups in forward/reverse order, conditioning on sampled commitments each round. All remaining deterministic positions can be predicted together on the next round. Exact support/fidelity, greedy rather than optimal round counts.",
            "recoverability_vs_completion": "Further self-generated commitments add no evidence about the hidden original. Refinement can improve support/TV, never exceed the observation's Bayes original-recovery ceiling. Exact joint posterior sampling is a separate 1-round oracle.",
            "work_proxies": "K*N and K*N^2 assume full canvas processing every pass; no model, vocabulary projection, decoder, hardware, or validation cost is measured.",
            "limits": "Tiny known priors, fixed public topology, two ASCII labels, no learning/generalization claims. Neither oracle rounds nor CPU runtime predict Mercury/GPU refinement steps or latency.",
        },
        "worlds": results,
        "exact_fusion_controls": run_fusion_experiments(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tokenizers", default="lexical,utf8_bytes")
    parser.add_argument("--samples-per-state", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20261020)
    args = parser.parse_args()
    report = run_experiments(tuple(args.tokenizers.split(",")), args.samples_per_state,
                             args.seed, progress=lambda message: print(message, flush=True))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    print(f"Wrote {args.output}; no neural inference or training.")


if __name__ == "__main__":
    main()

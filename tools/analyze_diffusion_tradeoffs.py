#!/usr/bin/env python3
"""Analyze saved token counts under explicit, idealized diffusion cost models.

No inference, training, network access, tokenizer loading, or new token counting
occurs. Prompt lengths are work proxies, not measured diffusion output lengths.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = ROOT / "benchmarks/results/initial-tokenizers.json"
BLOCK_SIZES = (32, 64, 128)
FORMATS = ("english", "json", "dsl", "prime", "m")
SCOPES = ("payload", "full_prompt")


def work_summary(lengths: list[int], block_size: int | None = None) -> dict:
    """Sum independent requests; square each length before aggregation."""
    if not lengths or any(type(n) is not int or n <= 0 for n in lengths):
        raise ValueError("Lengths must be a nonempty list of positive integers")
    result = {
        "requests": len(lengths),
        "tokens": sum(lengths),
        "sum_squared_lengths": sum(n * n for n in lengths),
    }
    if block_size is not None:
        if type(block_size) is not int or block_size <= 0:
            raise ValueError("Block size must be a positive integer")
        blocks = [(n + block_size - 1) // block_size for n in lengths]
        result.update(
            blocks=sum(blocks),
            padded_slots=sum(blocks) * block_size,
            unused_slots=sum(blocks) * block_size - sum(lengths),
        )
    return result


def comparison(candidate: list[int], baseline: list[int], ids: list[str]) -> dict:
    if len(candidate) != len(baseline) or len(candidate) != len(ids):
        raise ValueError("Candidate, baseline, and fixture IDs must be paired")
    c = work_summary(candidate)
    b = work_summary(baseline)
    ratios = [cn / bn for cn, bn in zip(candidate, baseline)]
    linear_speedup = b["tokens"] / c["tokens"]
    quadratic_speedup = b["sum_squared_lengths"] / c["sum_squared_lengths"]
    blocks = {}
    for size in BLOCK_SIZES:
        cb, bb = work_summary(candidate, size), work_summary(baseline, size)
        ccounts = [(n + size - 1) // size for n in candidate]
        bcounts = [(n + size - 1) // size for n in baseline]
        blocks[str(size)] = {
            "candidate_blocks": cb["blocks"],
            "baseline_blocks": bb["blocks"],
            "candidate_padded_slots": cb["padded_slots"],
            "baseline_padded_slots": bb["padded_slots"],
            "candidate_unused_slots": cb["unused_slots"],
            "baseline_unused_slots": bb["unused_slots"],
            "fewer_blocks_fixtures": sum(cn < bn for cn, bn in zip(ccounts, bcounts)),
            "equal_blocks_fixtures": sum(cn == bn for cn, bn in zip(ccounts, bcounts)),
            "more_blocks_fixtures": sum(cn > bn for cn, bn in zip(ccounts, bcounts)),
            "equal_steps_ideal_speedup": bb["blocks"] / cb["blocks"],
            "candidate_to_baseline_steps_break_even": bb["blocks"] / cb["blocks"],
        }
    return {
        "candidate_tokens": c["tokens"],
        "baseline_tokens": b["tokens"],
        "candidate_sum_squared_lengths": c["sum_squared_lengths"],
        "baseline_sum_squared_lengths": b["sum_squared_lengths"],
        "candidate_to_baseline_token_ratio": c["tokens"] / b["tokens"],
        "per_fixture_token_ratio_distribution": {
            "mean": statistics.mean(ratios),
            "median": statistics.median(ratios),
            "min": min(ratios),
            "max": max(ratios),
            "fewer_tokens_fixtures": sum(cn < bn for cn, bn in zip(candidate, baseline)),
            "equal_tokens_fixtures": sum(cn == bn for cn, bn in zip(candidate, baseline)),
            "more_tokens_fixtures": sum(cn > bn for cn, bn in zip(candidate, baseline)),
            "per_fixture": dict(zip(ids, ratios)),
        },
        "ideal_full_sequence": {
            "linear_equal_steps_speedup": linear_speedup,
            "quadratic_equal_steps_speedup": quadratic_speedup,
            "linear_candidate_to_baseline_steps_break_even": linear_speedup,
            "quadratic_candidate_to_baseline_steps_break_even": quadratic_speedup,
        },
        "ideal_fixed_block": blocks,
    }


def checked_rows(source: dict) -> tuple[dict, list[str], list[str]]:
    """Reject incomplete, unpaired, nonpositive, or wrong-schema inputs."""
    if source.get("schema") != "sylang-evaluation-v1" or source.get("semantic_schema") != "core-v0.1":
        raise ValueError("Expected saved core-v0.1 sylang-evaluation-v1 results")
    tokenizers = [item["id"] for item in source["tokenizers"]]
    if not tokenizers or len(tokenizers) != len(set(tokenizers)):
        raise ValueError("Tokenizer IDs must be present and unique")
    rows = {}
    for row in source["rows"]:
        key = (row["fixture_id"], row["format"])
        if key in rows or row["format"] not in FORMATS or row["semantic_roundtrip_exact"] is not True:
            raise ValueError("Duplicate, unsupported, or semantically invalid row")
        if set(row["tokenizers"]) != set(tokenizers):
            raise ValueError("Every row must have every tokenizer")
        for tokenizer in tokenizers:
            for scope in SCOPES:
                measurement = row["tokenizers"][tokenizer][scope]
                count = measurement["tokens"]
                if type(count) is not int or count <= 0:
                    raise ValueError("Token counts must be positive integers")
                if type(measurement["decoded_text_exact"]) is not bool:
                    raise ValueError("Text-fidelity diagnostics must be explicit booleans")
                if not all(isinstance(measurement[field], list) for field in ("special_token_ids", "unknown_token_ids")):
                    raise ValueError("Reserved/unknown-token diagnostics must be explicit lists")
        rows[key] = row
    ids = sorted({key[0] for key in rows})
    if not ids or len(ids) != source["fixture_manifest"]["count"]:
        raise ValueError("Fixture count differs from source manifest")
    if set(rows) != {(identifier, fmt) for identifier in ids for fmt in FORMATS}:
        raise ValueError("Every fixture must have all five formats")
    return rows, ids, tokenizers


def analyze(source: dict, source_bytes: bytes, source_name: str) -> dict:
    rows, ids, tokenizers = checked_rows(source)
    diagnostics = {}
    for identifier in ids:
        reasons = set()
        for fmt in FORMATS:
            for tokenizer in tokenizers:
                for scope in SCOPES:
                    value = rows[identifier, fmt]["tokenizers"][tokenizer][scope]
                    if not value["decoded_text_exact"]:
                        reasons.add("tokenizer_text_roundtrip_failure")
                    if value["special_token_ids"]:
                        reasons.add("registered_special_token_ids")
                    if value["unknown_token_ids"]:
                        reasons.add("unknown_token_ids")
        if reasons:
            diagnostics[identifier] = sorted(reasons)

    views = {}
    for name, selected in (
        ("all_fixtures_diagnostics_retained", ids),
        ("exclude_transport_diagnostics", [identifier for identifier in ids if identifier not in diagnostics]),
    ):
        if not selected:
            raise ValueError("No fixtures remain after diagnostic exclusions")
        per_tokenizer = {}
        for tokenizer in tokenizers:
            scopes = {}
            for scope in SCOPES:
                lengths = {
                    fmt: [rows[identifier, fmt]["tokenizers"][tokenizer][scope]["tokens"] for identifier in selected]
                    for fmt in FORMATS
                }
                scopes[scope] = {
                    "per_format_totals": {fmt: work_summary(lengths[fmt]) for fmt in FORMATS},
                    "m_vs_english": comparison(lengths["m"], lengths["english"], selected),
                    "m_vs_json": comparison(lengths["m"], lengths["json"], selected),
                }
            per_tokenizer[tokenizer] = scopes
        views[name] = {"fixture_count": len(selected), "fixture_ids": selected, "tokenizers": per_tokenizer}

    return {
        "schema": "sylang-diffusion-tradeoffs-v1",
        "source": {
            "path": source_name,
            "sha256": hashlib.sha256(source_bytes).hexdigest(),
            "bytes": len(source_bytes),
            "semantic_schema": source["semantic_schema"],
            "fixture_manifest": source["fixture_manifest"],
            "tokenizer_metadata": source["tokenizers"],
        },
        "methodology": {
            "new_model_execution": False,
            "new_token_measurement": False,
            "observations": "Counts are read from saved Qwen tokenizer measurements, not Mercury or another diffusion tokenizer.",
            "full_sequence_linear": "W=sum_i N_i; cost=K*W. Equal-step speedup=W_baseline/W_M.",
            "full_sequence_quadratic": "W=sum_i N_i^2 for separate fixture requests, never (sum_i N_i)^2; cost=K*W.",
            "step_break_even": "For the same K ratio on every fixture, M is faster only when K_M/K_baseline is strictly below the reported ratio. Equality is a tie; ratios below 1 require fewer M steps.",
            "fixed_block": "W=sum_i ceil(N_i/B). Each request is independently padded; cost=K*W*C(B), with the same cost per fixed block and steps per block. No prefix, partial-block, or cache model.",
            "block_sizes": list(BLOCK_SIZES),
            "critical_limit": "Payload and complete INPUT prompt lengths are hypothetical work proxies. Generated output lengths, denoising steps, output padding, cache behavior, quality, and measured diffusion latency are absent from the source.",
            "unknown_costs": "Ignores fixed overhead, bandwidth/compute mixture, vocabulary-head changes, translation, validation, retries, batching, hardware, and learned parallel dependencies. These idealized values are not measured speedups or universal upper bounds.",
            "exclusion_policy": "If any format/tokenizer/scope has text-fidelity failure, special IDs, or unknown IDs, exclude that fixture from every format, tokenizer, and scope in the second view. The first view retains all diagnostics.",
            "corpus_limit": "33 authored smoke fixtures and public holdouts do not establish general-language compression or trained-model performance.",
        },
        "transport_diagnostic_fixtures": diagnostics,
        "views": views,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, help="Write deterministic JSON to this path")
    args = parser.parse_args()
    source_bytes = args.source.read_bytes()
    try:
        source_name = str(args.source.resolve().relative_to(ROOT))
    except ValueError:
        source_name = str(args.source.resolve())
    report = analyze(json.loads(source_bytes), source_bytes, source_name)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    all_data = report["views"]["all_fixtures_diagnostics_retained"]
    print(f"Analyzed {all_data['fixture_count']} saved fixtures; no model execution or token counting.")
    for tokenizer, scopes in all_data["tokenizers"].items():
        for scope, value in scopes.items():
            comp = value["m_vs_english"]
            ideal = comp["ideal_full_sequence"]
            print(f"{tokenizer} {scope}: M/English tokens {comp['candidate_tokens']}/{comp['baseline_tokens']}; "
                  f"ideal equal-step linear/quadratic {ideal['linear_equal_steps_speedup']:.3f}x/"
                  f"{ideal['quadratic_equal_steps_speedup']:.3f}x")
    print("Diagnostic exclusions: " + ", ".join(report["transport_diagnostic_fixtures"]))
    if args.output:
        print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()

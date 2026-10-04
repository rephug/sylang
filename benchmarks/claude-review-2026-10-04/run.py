"""Reproduce every measurement cited by the 2026-10-04 independent review.

Offline and deterministic: no inference, no model weights, no network. Requires
the assets from fetch_assets.py plus tiktoken==0.14.0 and sentencepiece==0.2.2.

    python benchmarks/claude-review-2026-10-04/run.py --output benchmarks/claude-review-2026-10-04/results.json
"""
from __future__ import annotations

import argparse
import itertools
import json
import platform
import random
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path[:0] = [str(HERE), str(ROOT)]

import batch as B  # noqa: E402
import toks  # noqa: E402
import variants as V  # noqa: E402
from evaluation.harness import load_suite, prompt_for  # noqa: E402
from sylang_core import ASPECTS, EVIDENCES, FORMATS, POLARITIES, RELATIONS, TENSES, encode  # noqa: E402

STRESS = {"literal-max-length", "literal-special-token", "literal-decomposed", "literal-controls",
          "literal-unicode-separators", "literal-code", "literal-delimiters", "literal-quotes", "literal-emoji",
          "literal-cjk", "literal-arabic", "literal-composed", "literal-one-scalar"}
PAYLOAD_ORDER = ["english", "json", "dsl", "prime", "m", "english_concise", "m_body", "m_words", "m_opt",
                 "json_array", "json_short", "cdsl", "cdsl_full", "english_natural"]


def validate_qwen3(qwen, fixtures):
    committed = json.loads((ROOT / "benchmarks" / "results" / "initial-tokenizers.json").read_text(encoding="utf-8"))
    index = {(row["fixture_id"], row["format"]): row for row in committed["rows"]}
    compared = mismatched = fidelity_mismatched = 0
    for fx in fixtures:
        for fmt in FORMATS:
            payload = encode(fx["ast"], fmt)
            ref = index[(fx["id"], fmt)]["tokenizers"]["Qwen/Qwen3-0.6B-Base"]
            for scope, text in (("payload", payload), ("full_prompt", prompt_for(payload, fmt))):
                compared += 1
                mismatched += qwen.count(text) != ref[scope]["tokens"]
                fidelity_mismatched += qwen.roundtrip_exact(text) != ref[scope]["decoded_text_exact"]
    return dict(compared=compared, count_mismatches=mismatched, text_fidelity_mismatches=fidelity_mismatched)


def reversibility(fixtures):
    trees = [fx["ast"] for fx in fixtures]
    base = dict(kind="pred", subject="Mira", relation="see", object="lamp")
    for rel, pol, t, a, e in itertools.product(RELATIONS, POLARITIES, TENSES, ASPECTS, EVIDENCES):
        trees.append({"version": "core-v0.1", "statement": dict(base, relation=rel, polarity=pol, tense=t, aspect=a, evidence=e)})
    rng = random.Random(20261004)
    alphabet = 'abc XYZ"\\\n\x00é́中ع😀;()[]{},.:=+-/!if not see'

    def label():
        return "".join(rng.choice(alphabet) for _ in range(rng.randint(1, 24)))

    def tree(depth=1):
        if depth < 6 and rng.random() < 0.4:
            return dict(kind="if", condition=tree(depth + 1), consequence=tree(depth + 1))
        return dict(kind="pred", subject=label(), object=label(), relation=rng.choice(RELATIONS),
                    polarity=rng.choice(POLARITIES), tense=rng.choice(TENSES), aspect=rng.choice(ASPECTS),
                    evidence=rng.choice(EVIDENCES))

    trees += [{"version": "core-v0.1", "statement": tree()} for _ in range(2000)]
    failures = {}
    for name, render in V.RENDER.items():
        bad = 0
        for ast in trees:
            try:
                bad += V.DECODE[name](render(ast["statement"])) != ast
            except Exception:  # any decode error is a reversibility failure
                bad += 1
        failures[name] = bad
    return dict(trees=len(trees), seed=20261004, decode_failures=failures)


def payload_tables(T, fixtures):
    rend = {fx["id"]: V.all_renderings(fx["ast"]) for fx in fixtures}
    out = {}
    for subset, ids in (("all33", [f["id"] for f in fixtures]),
                        ("plain20_no_literal_stress", [f["id"] for f in fixtures if f["id"] not in STRESS])):
        table = {k: {t.name: sum(t.count(rend[i][k]) for i in ids) for t in T} for k in PAYLOAD_ORDER}
        table["_utf8_bytes"] = {k: sum(len(rend[i][k].encode("utf-8")) for i in ids) for k in PAYLOAD_ORDER}
        out[subset] = dict(fixtures=len(ids), tokens=table)
    q = next(t for t in T if t.name.startswith("qwen3"))
    out["qwen3_header_tokens_all33"] = dict(
        english=sum(q.count(encode(f["ast"], "english")) - q.count(encode(f["ast"], "english")[len("Sylang core-v0.1: "):]) for f in fixtures),
        m=sum(q.count(encode(f["ast"], "m")) - q.count(encode(f["ast"], "m")[len("M0.1:"):]) for f in fixtures))
    example = "time-past-progressive"
    out["example_pieces"] = {k: {t.name: t.pieces(rend[example][k]) for t in T if t.name in ("qwen3-reconstructed", "o200k_base")}
                             for k in ("english", "m", "m_words", "m_opt", "cdsl", "english_concise")}
    return out


def break_even(x, m):
    """N above which representation m is cheaper than x (None: never; 0: always)."""
    (xo, xp), (mo, mp) = x, m
    if mp < xp:
        return max(0.0, (mo - xo) / (xp - mp))
    if mp > xp:
        return None if mo >= xo else -((xo - mo) / (mp - xp))  # negative: m cheaper only below |N|
    return 0.0 if mo < xo else None


def batch_tables(T):
    out = {}
    for cname, labels, skew in (("en_skewed", B.SUBJECTS, True), ("en_uniform", B.SUBJECTS, False),
                                ("multilingual_skewed", B.MULTI, True)):
        trees = B.corpus(20261004, 200, labels, skew)
        for f in ("english_concise", "cdsl", "m_opt", "json_array"):
            assert all(V.DECODE[f](V.RENDER[f](a["statement"])) == a for a in trees), f
        corpus = {}
        for fname, (rf, leg) in B.FORMATS.items():
            corpus[fname] = {}
            for t in T:
                system = B.TASK + leg()
                row = {}
                for n in (1, 10, 100, 200):
                    block = B.records(rf, trees[:n])
                    row[str(n)] = dict(full=t.count(B.chat(t.name, system, "Facts:\n" + block + B.QUESTION)),
                                       overhead=t.count(B.chat(t.name, system, "Facts:\n" + B.QUESTION)),
                                       block=t.count(block))
                row["per_record_N200"] = row["200"]["block"] / 200
                corpus[fname][t.name] = row
        be = {}
        for t in T:
            m = corpus["m_repo"][t.name]
            for other in corpus:
                x = corpus[other][t.name]
                be.setdefault(other, {})[t.name] = break_even((x["200"]["overhead"], x["per_record_N200"]),
                                                              (m["200"]["overhead"], m["per_record_N200"]))
        out[cname] = dict(records=corpus, m_repo_break_even_N=be)
    flat = B.corpus(20261004, 200, B.SUBJECTS, True, cond_rate=0.0)

    def csv(trees, defaults):
        head = ("subject,relation,object,negated,tense,aspect,evidence (blank = positive/present/simple/unspecified)"
                if defaults else "subject,relation,object,polarity,tense,aspect,evidence")
        rows = []
        for a in trees:
            n = a["statement"]
            q = lambda s: json.dumps(s, ensure_ascii=False)  # noqa: E731
            if defaults:
                rows.append(",".join([q(n["subject"]), n["relation"], q(n["object"]),
                                      "not" if n["polarity"] == "negative" else "",
                                      "" if n["tense"] == "present" else n["tense"],
                                      "" if n["aspect"] == "simple" else n["aspect"],
                                      "" if n["evidence"] == "unspecified" else n["evidence"]]))
            else:
                rows.append(",".join([q(n["subject"]), n["relation"], q(n["object"]), n["polarity"], n["tense"],
                                      n["aspect"], n["evidence"]]))
        return head + "\n" + "\n".join(rows)

    blocks = {"csv_full": csv(flat, False), "csv_defaults": csv(flat, True),
              **{k: B.records(r, flat) for k, r in (("cdsl", "cdsl"), ("english_concise", "english_concise"),
                                                    ("m_repo", "m_repo"), ("m_opt", "m_opt"),
                                                    ("json_array", "json_array"), ("json_repo", "json_repo"),
                                                    ("english_natural*", "english_natural"))}}
    out["flat_predicates_N200_block_tokens"] = {k: {t.name: t.count(v) for t in T} for k, v in blocks.items()}
    return out


def probes(T):
    glyphs = {"ascii s": "s", "ascii word see": "see", "greek σ": "σ", "math ⊕": "⊕", "arrow →": "→",
              "logic ∴": "∴", "bracket ⟦": "⟦", "cjk 见": "见", "emoji 👁": "👁", "PUA U+E000": ""}
    base = {t.name: t.count('p("A",s,"B")\n' * 50) for t in T}
    glyph = {k: {t.name: (t.count(f'p("A",{v},"B")\n' * 50) - base[t.name]) / 50 for t in T} for k, v in glyphs.items()}
    unicode_probes = {"decomposed e+U+0301": '"café"', "Angstrom sign U+212B": '"Å"', "Ohm sign U+2126": '"Ω"',
                      "Kelvin sign U+212A": '"K"', "CJK compatibility U+F900": '"豈"', "ligature U+FB01": '"ﬁ"',
                      "fullwidth A U+FF21": '"Ａ"', "Hangul jamo sequence": '"가"', "BOM U+FEFF": '"a﻿b"',
                      "ZWJ emoji": '"👩‍🔬"', "escaped \\u0301": '"cafe\\u0301"'}
    uni = {k: {t.name: dict(exact=t.roundtrip_exact(v), tokens=t.count(v)) for t in T} for k, v in unicode_probes.items()}
    specials = {s: {t.name: dict(control=s in t.specials, tokens=t.count(s)) for t in T}
                for s in ["<|im_start|>", "<|endoftext|>", "<|eot_id|>", "<start_of_turn>", "[INST]", "<|start|>"]}
    numbers = {s: {t.name: t.count(s) for t in T} for s in ["7", "42", "2026", "3.14159", "1,250,000",
                                                             "12345678901234567890", "2026-10-04", "€19.99"]}
    return dict(glyph_extra_tokens_per_occurrence=glyph, unicode=uni, reserved_strings=specials, numbers=numbers)


def cost_model(batch):
    """Input-token-equivalent cost per call at N=100 under verified Anthropic multipliers (output 5x, cache read 0.1x)."""
    out = {}
    for tname in ("qwen3-reconstructed", "o200k_base", "gemma3"):
        rows = {}
        for fmt, data in batch["en_skewed"]["records"].items():
            r = data[tname]["100"]
            read_uncached = r["full"] + 5 * 3
            read_cached_legend = (r["full"] - r["overhead"]) + 0.1 * r["overhead"] + 5 * 3
            generate_uncached = r["overhead"] + 5 * r["block"]
            rows[fmt] = dict(read_uncached=read_uncached, read_cached_prefix=round(read_cached_legend, 1),
                             generate=generate_uncached)
        out[tname] = rows
    return dict(assumptions="output=5x input, cache read=0.1x input (Anthropic standard multipliers, verified 2026-10-04); "
                            "read answer=3 output tokens; generate = model emits all 100 records; cache write and "
                            "minimum cacheable length (512-4096 tokens) ignored; tokens from the named open tokenizer, "
                            "NOT Claude's (not public)", per_format=out)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    T = toks.load_all()
    fixtures, manifest = load_suite()
    batch = batch_tables([t for t in T if t.name != "cl100k_base"])
    result = dict(
        schema="sylang-claude-review-2026-10-04-v1", inference_run=False, training_run=False,
        environment=dict(python=sys.version.split()[0], platform=platform.platform()),
        fixture_manifest_sha256=manifest["sha256"],
        tokenizers=[t.meta() for t in T],
        qwen3_reconstruction_check=validate_qwen3(T[0], fixtures),
        reversibility=reversibility(fixtures),
        payload=payload_tables(T, fixtures),
        batch=batch,
        probes=probes(T),
        cost_model=cost_model(batch),
    )
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(json.dumps(dict(qwen3=result["qwen3_reconstruction_check"], reversibility=result["reversibility"]), indent=1))


if __name__ == "__main__":
    main()

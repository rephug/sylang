"""Rebuild this authored MIT fixture suite deterministically, without dependencies."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def pred(subject="Ada", relation="see", object="a bird", **changes):
    node = dict(kind="pred", subject=subject, relation=relation, object=object,
                polarity="positive", tense="present", aspect="simple",
                evidence="unspecified")
    node.update(changes)
    return node


def conditional(condition, consequence):
    return dict(kind="if", condition=condition, consequence=consequence)


def fixtures():
    rows = []

    def add(identifier, node, tags, split="development"):
        rows.append(dict(id=identifier, split=split, tags=tags,
                         ast=dict(version="core-v0.1", statement=node)))

    add("plain-positive", pred(), ["polarity", "latin"])
    add("plain-negative", pred(polarity="negative"), ["polarity", "latin"])
    for tense in ("past", "present", "future"):
        for aspect in ("simple", "progressive", "completed"):
            add(f"time-{tense}-{aspect}", pred("Bo", "help", "Cy", tense=tense,
                aspect=aspect), ["tense", "aspect", "latin"])
    for evidence in ("direct", "reported", "inferred"):
        add(f"evidence-{evidence}", pred("a box", "contain", "a key",
            evidence=evidence), ["evidence", "latin"])
    add("negated-condition", conditional(pred(polarity="negative"),
        pred("Ada", "help", "Bo", tense="future")), ["scope", "polarity", "conditional"])
    add("negated-consequence", conditional(pred(),
        pred("Ada", "help", "Bo", tense="future", polarity="negative")),
        ["scope", "polarity", "conditional"])
    add("nested-condition", conditional(conditional(pred(), pred("Bo", "see", "Cy")),
        pred("Cy", "help", "Ada", evidence="reported")), ["scope", "conditional"])
    add("nested-consequence", conditional(pred(), conditional(pred("Bo", "see", "Cy"),
        pred("Cy", "help", "Ada", evidence="reported"))), ["scope", "conditional"])
    add("literal-quotes", pred('a "quoted" name', object='backslash \\ and "quote"'),
        ["escaping", "latin"])
    add("literal-controls", pred("line\nnext\ttab", object="nul\x00carriage\rreturn"),
        ["escaping", "controls"])
    add("literal-cjk", pred("小林", "help", "小王"), ["literal-cjk"])
    add("literal-arabic", pred("ليلى", "see", "كتاب"), ["literal-arabic"])
    add("literal-emoji", pred("🧑🏽‍🔬", "see", "🦉"), ["unicode", "emoji"])
    add("literal-composed", pred("café", "contain", "é"), ["unicode", "normalization"])
    add("literal-decomposed", pred("cafe\u0301", "contain", "e\u0301"),
        ["unicode", "normalization"])
    add("literal-unicode-separators", pred("next\u0085line", "contain", "line\u2028paragraph\u2029end"),
        ["unicode", "escaping", "line-separators"])
    add("literal-code", pred("x = {'a': 1}", "contain", "</s>; DROP TABLE events;"),
        ["escaping", "code", "special-token-literal"])
    add("literal-special-token", pred("<|im_start|>", "contain", "[UNK]<|endoftext|>"),
        ["special-token-literal", "escaping"])
    add("literal-delimiters", pred("if;then|[{},]", "see", "Prime M DSL () : =>"),
        ["escaping", "delimiters"])
    add("literal-max-length", pred("a" * 256, "contain", "字" * 256), ["bounds"])
    add("literal-one-scalar", pred("a", "help", "b"), ["bounds"])
    add("holdout-negative-future", pred("the blue crate", "contain", "a spare bolt",
        polarity="negative", tense="future", aspect="completed", evidence="inferred"),
        ["polarity", "tense", "aspect", "evidence"], "holdout")
    add("holdout-conditional", conditional(pred("Mira", "help", "Noor", tense="past",
        aspect="progressive", evidence="direct"), pred("Noor", "see", "a comet",
        polarity="negative", tense="future", evidence="reported")),
        ["conditional", "scope", "polarity", "evidence"], "holdout")
    return rows


def main():
    rows = fixtures()
    data = "".join(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n"
                   for row in rows).encode("utf-8")
    (ROOT / "gold_suite.jsonl").write_bytes(data)
    manifest = dict(schema="sylang-fixtures-v1", file="gold_suite.jsonl",
                    sha256=hashlib.sha256(data).hexdigest(), bytes=len(data), count=len(rows),
                    license="MIT", provenance="Newly authored synthetic fixtures; no private documents or user data.",
                    generator="build_fixtures.py", semantic_schema="core-v0.1")
    (ROOT / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()

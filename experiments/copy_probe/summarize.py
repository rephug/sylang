"""Aggregate suite results (mean over seeds, with min-max) into Markdown tables.

    python summarize.py results/ > results/SUMMARY.md
"""

from __future__ import annotations

import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path


def _load(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _fmt(values: list[float], pct: bool = True) -> str:
    if not values:
        return "–"
    scale = 100 if pct else 1
    mean = statistics.fmean(values) * scale
    lo, hi = min(values) * scale, max(values) * scale
    digits = 0 if pct else 2
    if len(values) == 1 or abs(hi - lo) < 10 ** -digits:
        return f"{mean:.{digits}f}"
    return f"{mean:.{digits}f} ({lo:.{digits}f}–{hi:.{digits}f})"


def _positive_name(final: dict, suffix: str) -> str | None:
    for name in final:
        if name.endswith(suffix) and not name.startswith("none"):
            return name
    return None


def _steps_to(history: list[dict], key: str | None, threshold: float = 0.9) -> float | None:
    if key is None:
        return None
    for entry in history:
        if entry.get(key, 0) >= threshold:
            return entry["step"]
    return None


def _diag(final: dict, name: str | None, *path: str) -> float | None:
    node = final.get(name or "", {})
    for key in path:
        if not isinstance(node, dict) or key not in node:
            return None
        node = node[key]
    return node


def _table(headers: list[str], rows: list[list[str]]) -> str:
    out = ["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|"]
    out += ["| " + " | ".join(row) + " |" for row in rows]
    return "\n".join(out)


def replica(results: list[dict], keys: tuple[str, ...]) -> str:
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for r in results:
        groups[tuple(r["overrides"].get(k, 1.0) for k in keys)].append(r)
    rows = []
    for key in sorted(groups):
        runs = groups[key]

        def col(name, *path, pct=True):
            vals = [_diag(r["final"], name, *path) for r in runs]
            return _fmt([v for v in vals if v is not None], pct)

        rows.append([
            *(str(k) for k in key),
            str(len(runs)),
            col("select_seen", "exact"),
            col("select_unseen", "exact"),
            col("none", "exact"),
            col("none_length_matched", "exact"),
            col("select_unseen", "mode_fraction"),
            col("select_unseen", "teacher_forced", "id_first", "acc"),
            col("select_unseen", "teacher_forced", "id_rest", "acc"),
            col("select_unseen", "diagnostics", "prompt_dependence_js_bits", pct=False),
            col("select_unseen", "diagnostics", "swap", "mean_tv", pct=False),
            col("select_unseen", "diagnostics", "replace", "mean_tv", pct=False),
        ])
    headers = [*keys, "seeds", "seen pair %", "unseen pair %", "empty %", "len-matched empty %",
               "mode share %", "first-byte acc %", "later-byte acc %", "JS bits", "swap TV", "replace TV"]
    return _table(headers, rows)


def matrix(results: list[dict]) -> str:
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for r in results:
        o = r["overrides"]
        ids = "replica pool" if o["task"] == "replica" else o["ids"]
        groups[(o["task"], ids, o.get("jitter", 0), o["batch_size"], o["lr"], o["steps"])].append(r)
    rows = []
    for key in sorted(groups):
        runs = groups[key]
        task = key[0]
        if task == "replica":
            train_name, held_name, none_name = "select_seen", "select_unseen", "none_length_matched"
        else:
            train_name = _positive_name(runs[0]["final"], "_train_ids")
            held_name = _positive_name(runs[0]["final"], "_heldout_ids")
            none_name = "none_heldout_ids"

        def col(name, *path, pct=True):
            vals = [_diag(r["final"], name, *path) for r in runs]
            return _fmt([v for v in vals if v is not None], pct)

        reach = [_steps_to(r["history"], held_name) for r in runs]
        reached = [s for s in reach if s is not None]
        rows.append([
            *(str(k) for k in key),
            str(len(runs)),
            col(train_name, "exact"),
            col(held_name, "exact"),
            col(none_name, "exact"),
            col(held_name, "teacher_forced", "id_first", "acc"),
            col(held_name, "diagnostics", "prompt_dependence_js_bits", pct=False),
            col(held_name, "diagnostics", "swap", "mean_tv", pct=False),
            col(held_name, "diagnostics", "replace", "mean_tv", pct=False),
            f"{len(reached)}/{len(runs)}" + (f" (median {statistics.median(reached):.0f})" if reached else ""),
            _fmt([r["train_seconds"] for r in runs], pct=False),
        ])
    headers = ["task", "ids", "jitter", "batch", "lr", "updates", "seeds", "train-id exact %", "held-out exact %",
               "held-out empty % (replica: length-matched)", "held-out first-byte %", "JS bits", "swap TV", "replace TV",
               "runs ≥90% held-out (step)", "train s"]
    return _table(headers, rows)


def controls(results: list[dict]) -> str:
    rows = []
    for r in sorted(results, key=lambda r: r["overrides"]["seed"]):
        reach = _steps_to(r["history"], "repeat")
        rows.append([str(r["overrides"]["seed"]), str(r["overrides"]["steps"]), _fmt([r["final"]["repeat"]["exact"]]),
                     str(reach), str(r["train_seconds"])])
    return _table(["seed", "updates", "held-out exact %", "first step ≥90%", "train s"], rows)


def real_pipeline(results: list[dict]) -> str:
    """Runs of the project's own tiny model and data (real_pipeline.py)."""
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for r in results:
        o = r["overrides"]
        groups[(o["task"], o["data"], o["batch"], o["lr"], o["updates"])].append(r)
    sections = []
    for task, cohorts, probe in (
        ("copy", ["seen_copy", "seen_none", "heldout_copy", "heldout_none"], "heldout_copy"),
        ("binding", ["seen_selected", "seen_empty", "unseen_pair_selected", "unseen_pair_empty",
                     "fresh_id_selected", "fresh_id_empty"], "unseen_pair_selected"),
    ):
        rows = []
        for key in sorted(k for k in groups if k[0] == task):
            runs = groups[key]

            def col(name, field, pct=True):
                vals = [r["final"][name][field] for r in runs if field in r["final"].get(name, {})]
                return _fmt(vals, pct)

            rows.append([*(str(k) for k in key[1:]), str(len(runs)), *(col(c, "exact") for c in cohorts),
                         col(probe, "first_byte_acc"), col(probe, "first_byte_js_bits", pct=False),
                         *( [col(probe, "swap_tv", pct=False)] if task == "binding" else [] ),
                         _fmt([r["train_seconds"] for r in runs], pct=False)])
        headers = ["data", "batch", "lr", "updates", "seeds", *(f"{c} %" for c in cohorts),
                   f"{probe} first-byte %", "JS bits", *(["swap TV"] if task == "binding" else []), "train s"]
        sections.append(f"### {task}\n\n" + _table(headers, rows))
    return "\n\n".join(sections)


def copy_matrix(results: list[dict]) -> str:
    """COPY initialisation x identifier breadth x exposure (followup.py)."""
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for r in results:
        o = r["overrides"]
        groups[(o["init"], o["breadth"], o["updates"])].append(r)
    rows = []
    for key in sorted(groups):
        runs = groups[key]

        def frac(cohort, field):
            return _fmt([r["final"][cohort][field] / r["final"][cohort]["n"] for r in runs])

        rows.append([*(str(k) for k in key), str(runs[0]["exposures_per_row"]), str(len(runs)),
                     frac("seen", "copy_exact"), frac("heldout_legacy12", "copy_exact"),
                     frac("heldout_fresh108", "copy_exact"),
                     _fmt([(r["final"]["heldout_legacy12"]["none_exact"] + r["final"]["heldout_fresh108"]["none_exact"]) / 120
                           for r in runs]),
                     _fmt([r["final"]["heldout_legacy12"]["wrong_that_are_training_ids"]
                           + r["final"]["heldout_fresh108"]["wrong_that_are_training_ids"] for r in runs], pct=False)])
    headers = ["init", "train ids", "updates", "exposures/row", "seeds", "seen COPY %", "legacy-12 held-out COPY %",
               "fresh-108 held-out COPY %", "held-out NONE %", "held-out errors that are training ids"]
    return _table(headers, rows)


def _first_reaching(history: list[dict], field: str, threshold: float = 0.9):
    for h in history:
        if h["iid"]["exact"] >= threshold:
            return h[field]
    return None


def language(results: list[dict]) -> str:
    """English-vs-Sylang comparison (language.py)."""
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for r in results:
        o = r["overrides"]
        arm = o["arm"] + (f' ({o["position"]})' if "position" in o else "")
        groups[(arm, o["init"], "".join(map(str, o["k"])), o["batch"], o["lr"], o["updates"])].append(r)
    order = {"english": 0, "english_natural": 1, "words": 2, "codes": 3, "codes_eqlen": 4, "codes_jitter": 5}
    rows, curve, curve_steps = [], [], []
    for key in sorted(groups, key=lambda k: (k[1], k[2], k[3], k[5], order.get(k[0].split(" ")[0], 9), k[0], k[4])):
        runs = groups[key]
        last = [r["history"][-1] for r in runs]

        def col(panel, field="exact", pct=True):
            vals = [h[panel][field] for h in last if panel in h and h[panel].get(field) is not None]
            return _fmt(vals, pct)

        reach_items = [_first_reaching(r["history"], "items_seen") for r in runs]
        reach_bytes = [_first_reaching(r["history"], "bytes_seen") for r in runs]
        ok_items = [x for x in reach_items if x is not None]
        ok_bytes = [x for x in reach_bytes if x is not None]
        rows.append([*(str(k) for k in key), f"{statistics.median(h['step'] for h in last):,.0f}", str(len(runs)),
                     col("iid"), col("iid", "selected"), col("iid", "always_empty_baseline"), col("dev"),
                     col("held_out_combo"), col("swap"), col("flip"),
                     col("cue_conflict") if any("cue_conflict" in h for h in last) else "–",
                     _fmt([h["swap_margin"]["positive"] for h in last if "swap_margin" in h]),
                     f"{len(ok_items)}/{len(runs)}" + (f" ({statistics.median(ok_items):,.0f})" if ok_items else ""),
                     (f"{statistics.median(ok_bytes) / 1e6:.2f}" if ok_bytes else "–"),
                     f"{statistics.fmean(r['mean_prompt_bytes'] for r in runs):.0f}" if "mean_prompt_bytes" in runs[0] else "–",
                     _fmt([h["train_seconds"] for h in last], pct=False)])
        curve.append((key, runs))
    headers = ["arm", "init", "K", "batch", "lr", "updates", "trained to (median)", "seeds", "held-out ids %", "selected %",
               "always-[] baseline %", "dev %", "held-out combo %", "swap %", "flip %", "cue-conflict %",
               "swap margin > 0 %", "runs ≥90% (median items)", "median MB to 90%", "prompt bytes (trained)",
               "train s (shared CPU)"]
    table = _table(headers, rows)
    steps = sorted({h["step"] for _, runs in curve for r in runs for h in r["history"]})
    for (panel, field, pct), title in (
            (("iid", "exact", True), "Held-out-identifier exact accuracy"),
            (("iid", "selected", True), "Held-out-identifier exact accuracy on selected (non-empty) answers"),
            (("swap_margin", "mean", False), "Swap margin, mean nats (> 0: the answer follows the roles)")):
        lines = []
        for key, runs in curve:
            by_step = {s: [h[panel][field] for r in runs for h in r["history"]
                           if h["step"] == s and h[panel].get(field) is not None] for s in steps}
            lines.append([key[0], key[1], key[2], f"{key[4]:g}",
                          *(_fmt(by_step[s], pct) if by_step[s] else "–" for s in steps)])
        if lines:
            table += f"\n\n{title} by update (runs that stopped early at the ceiling end sooner):\n\n" + _table(
                ["arm", "init", "K", "lr", *(str(s) for s in steps)], lines)
    return table


def language_benchmark(report: dict) -> str:
    """Serial single-thread inference cost per query (language.py benchmark)."""
    base = report.get("english", {})
    rows = []
    for arm, r in report.items():
        rel = (f"{r['prefill_gflops_mean'] / base['prefill_gflops_mean']:.2f}"
               if base.get("prefill_gflops_mean") else "–")
        rows.append([arm, f"{r['mean_prompt_bytes']:.1f}",
                     " / ".join(f"{r['mean_prompt_bytes_by_k'][str(k)]:.0f}" for k in (1, 2, 3, 4)),
                     f"{r['mean_answer_bytes']:.1f}", f"{r['prefill_ms_median']:.3f}", f"{r['decode_ms_mean']:.2f}",
                     f"{r['prefill_gflops_mean']:.4f}", f"{r['decode_gflops_mean']:.4f}", rel])
    return _table(["arm", "prompt bytes (natural)", "by K = 1 / 2 / 3 / 4", "answer bytes", "prefill ms (median)",
                   "decode ms (mean)", "prefill GFLOPs", "decode GFLOPs", "prefill FLOPs vs english"], rows)


def _peak(run: dict, panel: str = "iid", field: str = "exact") -> float:
    return max(h[panel][field] for h in run["history"])


def study(results: list[dict]) -> str:
    """Design-v2 study (language.py --suite study): K up to 8, hold-outs separated."""
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for r in results:
        o = r["overrides"]
        groups[(o["arm"], o.get("position", "learned"), o["init"], o.get("curriculum", "mixed"),
                "".join(map(str, o["k"])), o["updates"], o["lr"], o["batch"])].append(r)
    order = {"english_natural": 0, "codes": 1, "compact": 2}
    rows, curve = [], []
    panels_seen: list[str] = []
    for key in sorted(groups, key=lambda k: (k[4], k[1], k[3], k[2], order.get(k[0], 9), k[0])):
        runs = groups[key]
        last = [r["history"][-1] for r in runs]
        for h in last:
            for name, v in h.items():
                if isinstance(v, dict) and "exact" in v and name not in panels_seen:
                    panels_seen.append(name)

        def col(panel, field="exact", pct=True):
            vals = [h[panel][field] for h in last if panel in h and h[panel].get(field) is not None]
            return _fmt(vals, pct) if vals else "–"

        reach_items = [x for x in (_first_reaching(r["history"], "items_seen") for r in runs) if x is not None]
        reach_bytes = [x for x in (_first_reaching(r["history"], "bytes_seen") for r in runs) if x is not None]
        long_k = [statistics.fmean(v for k, v in h["iid"]["by_k"].items() if int(k) >= 5)
                  for h in last if any(int(k) >= 5 for k in h["iid"]["by_k"])]
        switches = [h.get("stage_switch") for r in runs for h in r["history"] if h.get("stage_switch")]
        rows.append([key[0], key[1], key[2], key[3] + (f" (switch {statistics.median(switches):,.0f})" if switches else ""),
                     str(len(runs)), f"{statistics.median(h['step'] for h in last):,.0f}",
                     f"{len(reach_items)}/{len(runs)}" + (f" ({statistics.median(reach_items):,.0f}" + (
                         f"; {min(reach_items):,.0f}–{max(reach_items):,.0f})" if len(reach_items) > 1 else ")") if reach_items else ""),
                     (f"{statistics.median(reach_bytes) / 1e6:.1f}" if reach_bytes else "–"),
                     _fmt([_peak(r) for r in runs]), col("iid"), col("iid", "selected"), col("iid", "empty"),
                     _fmt(long_k) if long_k else "–",
                     col("combo", "selected"), col("both", "selected"), col("similar", "selected"),
                     col("reorder"), col("swap"), col("flip"),
                     f"{statistics.fmean(r['mean_prompt_bytes'] for r in runs):.0f}",
                     _fmt([h["train_seconds"] for h in last], pct=False)])
        curve.append((key, runs))
    headers = ["arm", "positions", "init", "curriculum", "seeds", "trained to (median)",
               "runs ≥90% iid (median items; range)", "median MB to 90%", "peak iid %", "final iid %",
               "final iid selected %", "final iid empty %", "iid K=5–8 %", "combo selected %",
               "both selected %", "similar selected %", "reorder %", "swap %", "flip %",
               "prompt bytes", "train s (shared CPU)"]
    table = _table(headers, rows)
    steps = sorted({h["step"] for _, runs in curve for r in runs for h in r["history"]})
    for (panel, field, pct), title in ((("iid", "exact", True), "iid exact"),
                                        (("iid", "selected", True), "iid selected-only exact"),
                                        (("combo", "selected", True), "combo selected-only exact")):
        lines = []
        for key, runs in curve:
            by_step = {s: [h[panel][field] for r in runs for h in r["history"]
                           if h["step"] == s and panel in h and h[panel].get(field) is not None] for s in steps}
            lines.append([key[0], key[1], key[2], key[3], *(_fmt(by_step[s], pct) if by_step[s] else "–" for s in steps)])
        table += f"\n\n{title} by update (early-stopped runs end sooner):\n\n" + _table(
            ["arm", "positions", "init", "curriculum", *(str(s) for s in steps)], lines)
    return table


def main() -> None:
    root = Path(sys.argv[1] if len(sys.argv) > 1 else "results")
    sections = []
    for name, render in (
        ("replica_sweep", lambda r: replica(r, ("lr", "batch_size"))),
        ("replica_sweep_v0", lambda r: replica(r, ("lr", "batch_size"))),
        ("replica_budget", lambda r: replica(r, ("lr", "steps", "first_byte_weight"))),
        ("controls", controls),
        ("matrix", matrix),
        ("batch1_copy", matrix),
        ("balanced12", matrix),
        ("real_pipeline", real_pipeline),
        ("copy_matrix", copy_matrix),
        ("language_lr", language),
        ("language", language),
        ("language_k1_u4000_seed0", language),
        ("language_k1", language),
        ("language_alibi", language),
        ("language_alibi_main", language),
        ("language_copy96", language),
        ("selection_init", language),
    ):
        path = root / f"{name}.jsonl"
        if path.exists():
            sections.append(f"## {name}\n\n{render(_load(path))}\n")
    study_files = sorted(root.glob("language_study*.jsonl"))
    if study_files:
        merged = [r for f in study_files for r in _load(f)]
        sections.append(f"## language_study (merged from {', '.join(f.name for f in study_files)})\n\n{study(merged)}\n")
    bench = root / "language_benchmark.json"
    if bench.exists():
        sections.append(f"## language_benchmark\n\n{language_benchmark(json.loads(bench.read_text()))}\n")
    print("# Copy-probe results\n")
    print("Generated by `python summarize.py results`. Mean over seeds, with min–max in brackets.")
    print("For the replica, 'train-id' means the 8 seen selected checks and 'held-out' the 16 unseen ones.")
    print("JS bits: prompt dependence at the answer start. TV: total-variation distance after swapping")
    print("subject/object (swap) or renaming the first target identifier (replace).")
    print("Caution: the matrix 'pool12' arms are confounded. Their held-out identifiers contain characters")
    print("never emitted at that position in training (10-12 of 12), so held-out failure is guaranteed;")
    print("'balanced12' uses the project's balanced identifiers instead.\n")
    print("\n".join(sections))


if __name__ == "__main__":
    main()

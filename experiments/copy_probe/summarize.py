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
        ks = sorted(int(k) for k in r['mean_prompt_bytes_by_k'])
        decode_ms = r.get('decode_ms_mean_reference_decoder', r.get('decode_ms_mean'))
        decode_gf = r.get('decode_gflops_mean_reference_decoder', r.get('decode_gflops_mean'))
        rows.append([arm, f"{r['mean_prompt_bytes']:.1f}",
                     " / ".join(f"{r['mean_prompt_bytes_by_k'][str(k)]:.0f}" for k in ks),
                     f"{r['mean_answer_bytes']:.1f}", f"{r['prefill_ms_median']:.3f}", f"{decode_ms:.2f}",
                     f"{r['prefill_gflops_mean']:.4f}", f"{decode_gf:.4f}",
                     f"{r['decode_gflops_mean_kv_cached']:.5f}" if 'decode_gflops_mean_kv_cached' in r else "–", rel])
    return _table(["arm", "prompt bytes (natural)", "by K", "answer bytes", "prefill ms (median)",
                   "decode ms (reference decoder, mean)", "prefill GFLOPs", "decode GFLOPs (reference decoder)",
                   "decode GFLOPs (KV-cached, analytic)", "prefill FLOPs vs english"], rows)


def _peak(run: dict, panel: str = "iid", field: str = "exact") -> float:
    return max(h[panel][field] for h in run["history"])


def _selected_checkpoint(run: dict) -> dict:
    """The dev-selected checkpoint: the earliest grid checkpoint with the highest dev exact."""
    step = run.get("dev_selected_step")
    for h in run["history"]:
        if h["step"] == step:
            return h
    return max(run["history"], key=lambda h: (h["dev"]["exact"], -h["step"]))


def _items_to_90(run: dict, field: str = "items_seen", threshold: float = 0.9):
    """First point on the fine iid curve (grid checkpoints included) at or above the
    threshold; None when never reached (reported as censored)."""
    points = [(h["step"], h[field], h["iid"]["exact"]) for h in run["history"]]
    points += [(f["step"], f[field], f["iid_exact"]) for f in run.get("fine", [])]
    for _, value, exact in sorted(points):
        if exact >= threshold:
            return value
    return None


def _median_censored(values: list) -> str:
    ok = [v for v in values if v is not None]
    censored = len(values) - len(ok)
    if not ok or censored > len(values) // 2:
        return f"∞ ({censored}/{len(values)} censored)"
    med = statistics.median(ok) if censored == 0 else sorted(ok)[len(values) // 2] if len(ok) > len(values) // 2 else None
    text = f"{med:,.0f}" if med is not None else "∞"
    if len(ok) > 1:
        text += f" ({min(ok):,.0f}–{max(ok):,.0f})"
    if censored:
        text += f" [{censored} censored]"
    return text


def _paired_bootstrap(a_flags: list[str], b_flags: list[str], resamples: int = 1000, seed: int = 0):
    """Mean of per-item differences a-b (in points) averaged over seeds, with a 90% bootstrap
    interval over items. a_flags/b_flags: one flag string per seed, same panel items."""
    import random
    pairs = list(zip(a_flags, b_flags))
    n = len(pairs[0][0])
    diffs = [statistics.fmean(int(a[i]) - int(b[i]) for a, b in pairs) for i in range(n)]
    rng = random.Random(seed)
    means = sorted(statistics.fmean(diffs[rng.randrange(n)] for _ in range(n)) for _ in range(resamples))
    return 100 * statistics.fmean(diffs), 100 * means[int(0.05 * resamples)], 100 * means[int(0.95 * resamples) - 1]


def study(results: list[dict]) -> str:
    """Design-v2 study (language.py --suite study): K up to 8, hold-outs separated.
    Headline numbers are read at the dev-selected checkpoint."""
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for r in results:
        o = r["overrides"]
        groups[(o["arm"], o.get("position", "learned"), o["init"], o.get("curriculum", "mixed"),
                "".join(map(str, o["k"])), o["updates"], o["lr"], o["batch"])].append(r)
    order = {"english_natural": 0, "codes": 1, "compact": 2}
    rows, curve = [], []
    for key in sorted(groups, key=lambda k: (k[4], k[1], k[3], k[2], order.get(k[0], 9), k[0])):
        runs = sorted(groups[key], key=lambda r: r["overrides"]["seed"])
        sel = [_selected_checkpoint(r) for r in runs]
        last = [r["history"][-1] for r in runs]

        def col(panel, field="exact", pct=True, at=sel):
            vals = [h[panel][field] for h in at if panel in h and h[panel].get(field) is not None]
            return _fmt(vals, pct) if vals else "–"

        long_k = [statistics.fmean(v for k, v in h["iid"]["by_k"].items() if int(k) >= 5)
                  for h in sel if any(int(k) >= 5 for k in h["iid"]["by_k"])]
        switches = [h.get("stage_switch") for r in runs for h in r["history"] if h.get("stage_switch")]
        rows.append([key[0], key[1], key[2], key[3] + (f" (switch {statistics.median(switches):,.0f})" if switches else ""),
                     ",".join(str(r["overrides"]["seed"]) for r in runs),
                     f"{statistics.median(r['dev_selected_step'] or 0 for r in runs):,.0f}",
                     f"{statistics.median(h['step'] for h in last):,.0f}",
                     _median_censored([_items_to_90(r) for r in runs]),
                     _median_censored([(lambda v: v / 1e6 if v else None)(_items_to_90(r, "bytes_seen")) for r in runs]).replace(",", ""),
                     col("iid"), col("iid", "selected"), col("iid", "empty"), _fmt(long_k) if long_k else "–",
                     col("seen", "selected"), col("combo", "selected"), col("both", "selected"),
                     col("similar", "selected"), col("similar_control", "selected"),
                     col("reorder"), col("swap"), col("flip"), _fmt([_peak(r) for r in runs]),
                     f"{statistics.fmean(r['mean_prompt_bytes'] for r in runs):.0f}",
                     _fmt([h["train_seconds"] for h in last], pct=False)])
        curve.append((key, runs))
    headers = ["arm", "positions", "init", "curriculum", "seeds", "dev-selected update (median)",
               "trained to (median)", "items to 90% iid (median; range)", "MB to 90% iid",
               "iid %", "iid selected %", "iid empty %", "iid K=5–8 %", "seen selected %", "combo selected %",
               "both selected %", "similar selected %", "similar-control selected %", "reorder %", "swap %",
               "flip %", "peak iid % (descriptive)", "prompt bytes", "train s (shared CPU)"]
    table = "At the dev-selected checkpoint unless stated. Mean over seeds (min–max).\n\n" + _table(headers, rows)
    # Per-seed ratios of items-to-90% for every method against the mixed/fresh baseline of the same arm.
    ratio_rows = []
    for key, runs in curve:
        base_key = (key[0], key[1], "fresh", "mixed", *key[4:])
        if key[2:4] == ("fresh", "mixed") or base_key not in groups:
            continue
        base = {r["overrides"]["seed"]: _items_to_90(r) for r in groups[base_key]}
        ratios = []
        for r in runs:
            m, b = _items_to_90(r), base.get(r["overrides"]["seed"])
            ratios.append("– (both censored)" if b is None and m is None else "∞ (method censored)" if m is None
                          else "< 1 (baseline censored)" if b is None else f"{m / b:.2f}")
        ratio_rows.append([key[0], key[1], key[2], key[3], ", ".join(ratios)])
    if ratio_rows:
        table += "\n\nItems-to-90% ratio, method / baseline (fresh, mixed), paired by seed:\n\n" + _table(
            ["arm", "positions", "init", "curriculum", "ratio per seed"], ratio_rows)
    # Paired per-item bootstrap: compact minus codes on the shared panels, averaged over seeds.
    boot_rows = []
    for key, runs in curve:
        if key[0] != "compact":
            continue
        codes_key = ("codes", *key[1:])
        if codes_key not in groups:
            continue
        codes_runs = {r["overrides"]["seed"]: r for r in groups[codes_key]}
        shared = [r for r in runs if r["overrides"]["seed"] in codes_runs]
        if not shared:
            continue
        for panel in ("iid", "combo", "both", "similar", "reorder"):
            a = [_selected_checkpoint(r)[panel]["flags"] for r in shared]
            b = [_selected_checkpoint(codes_runs[r["overrides"]["seed"]])[panel]["flags"] for r in shared]
            mean, lo, hi = _paired_bootstrap(a, b)
            boot_rows.append([key[1], key[2], key[3], panel, str(len(shared)), f"{mean:+.1f}", f"[{lo:+.1f}, {hi:+.1f}]",
                              "yes" if lo > -3 else "no"])
    if boot_rows:
        table += ("\n\ncompact − codes, paired per item at the dev-selected checkpoint, points (90% bootstrap over "
                  "items; non-inferior if the lower bound is above −3):\n\n"
                  + _table(["positions", "init", "curriculum", "panel", "seeds", "mean", "90% interval", "non-inferior"], boot_rows))
    steps = sorted({h["step"] for _, runs in curve for r in runs for h in r["history"]})
    for (panel, field, pct), title in ((("iid", "exact", True), "iid exact"),
                                        (("iid", "selected", True), "iid selected-only exact"),
                                        (("combo", "selected", True), "combo selected-only exact"),
                                        (("both", "selected", True), "both selected-only exact")):
        lines = []
        for key, runs in curve:
            by_step = {s: [h[panel][field] for r in runs for h in r["history"]
                           if h["step"] == s and panel in h and h[panel].get(field) is not None] for s in steps}
            lines.append([key[0], key[1], key[2], key[3], *(_fmt(by_step[s], pct) if by_step[s] else "–" for s in steps)])
        table += f"\n\n{title} by grid update (early-stopped runs end sooner):\n\n" + _table(
            ["arm", "positions", "init", "curriculum", *(str(s) for s in steps)], lines)
    return table


def study3(results: list[dict]) -> str:
    """Design-v3 study (language.py --suite study3): K up to 12 in training, variable identifiers,
    varied wording, JSON/CSV comparators, a fully held-out panel. Dev-selected checkpoint."""
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for r in results:
        o = r["overrides"]
        groups[(o["arm"], o.get("corpus", "all"), o.get("position", "learned"), o["init"],
                o.get("curriculum", "mixed"), o["updates"], o["lr"], o["batch"])].append(r)
    # The §10.1 table reads the final checkpoint for the pilot's fixed-shape corpora (deviation 2); here every
    # group is read at the dev-selected checkpoint, which for those corpora is on out-of-distribution panels.
    order = {"english": 0, "sylang2": 1, "sylang1": 2, "json": 3, "csv": 4}
    rows, curve = [], []
    for key in sorted(groups, key=lambda k: (k[1], k[2], k[4], k[3], order.get(k[0], 9), k[0])):
        runs = sorted(groups[key], key=lambda r: r["overrides"]["seed"])
        sel = [_selected_checkpoint(r) for r in runs]
        last = [r["history"][-1] for r in runs]

        def col(panel, field="exact", pct=True, at=sel):
            vals = [h[panel][field] for h in at if panel in h and h[panel].get(field) is not None]
            return _fmt(vals, pct) if vals else "–"

        def by_len(panel, length):
            vals = [h[panel]["by_len"][str(length)] for h in sel
                    if panel in h and str(length) in h[panel].get("by_len", {})]
            return _fmt(vals) if vals else "–"

        rows.append([key[0], key[1], key[2], key[4],
                     ",".join(str(r["overrides"]["seed"]) for r in runs),
                     f"{statistics.median(r['dev_selected_step'] or 0 for r in runs):,.0f}",
                     f"{statistics.median(h['step'] for h in last):,.0f}",
                     _median_censored([_items_to_90(r) for r in runs]),
                     col("iid"), col("iid", "selected"), col("wording"), col("wording", "selected"),
                     col("combo", "selected"), col("long"), col("long", "selected"),
                     col("full"), col("full", "selected"), col("longer"),
                     col("reorder"), col("swap"), col("flip"),
                     by_len("iid", 2), by_len("iid", 5),
                     f"{statistics.fmean(r['mean_prompt_bytes'] for r in runs):.0f}",
                     _fmt([h["train_seconds"] for h in last], pct=False)])
        curve.append((key, runs))
    headers = ["arm", "corpus", "positions", "curriculum", "seeds", "dev-selected update (median)",
               "trained to (median)", "items to 90% iid (median; range)",
               "iid %", "iid selected %", "wording %", "wording selected %", "combo selected %",
               "long %", "long selected %", "full %", "full selected %", "longer (K 17–20) %", "reorder %", "swap %", "flip %",
               "iid longest-id 2 %", "iid longest-id 5 %", "prompt bytes", "train s (shared CPU)"]
    table = ("At the dev-selected checkpoint unless stated. Mean over seeds (min–max). Panels: iid = trained "
             "wording/combinations, K 1..12; wording = held-out wording; combo = a held-out combination; long = "
             "K 13..16; full = held-out wording + held-out combination + K 13..16.\n\n" + _table(headers, rows))
    # Secondary reading (pre-registered as a labelled extra after the escape rate turned out seed-dependent):
    # the learned seeds only, i.e. seeds whose fine iid curve reached 90%.
    learned_rows = []
    for key, runs in curve:
        learned = [r for r in runs if _items_to_90(r) is not None]
        if not learned:
            learned_rows.append([key[0], key[1], f"0/{len(runs)}", *(["–"] * 9)])
            continue
        sel_l = [_selected_checkpoint(r) for r in learned]

        def lcol(panel, field="exact"):
            vals = [h[panel][field] for h in sel_l if panel in h and h[panel].get(field) is not None]
            return _fmt(vals) if vals else "–"

        learned_rows.append([key[0], key[1], f"{len(learned)}/{len(runs)}",
                             _median_censored([_items_to_90(r) for r in learned]),
                             lcol("iid"), lcol("wording"), lcol("combo", "selected"), lcol("long"), lcol("full"),
                             lcol("longer"), lcol("reorder"), lcol("swap")])
    table += ("\n\nLearned seeds only (fine iid curve reached 90%), dev-selected checkpoint, mean (min–max); "
              "the escape rate is learned/seeds:\n\n"
              + _table(["arm", "corpus", "learned/seeds", "items to 90% iid", "iid %", "wording %", "combo selected %",
                        "long %", "full %", "longer %", "reorder %", "swap %"], learned_rows))
    # Paired per-item bootstrap: every arm minus english, on the shared panels, averaged over shared seeds.
    boot_rows = []
    for key, runs in curve:
        if key[0] == "english":
            continue
        base_key = ("english", *key[1:])
        if base_key not in groups:
            continue
        base_runs = {r["overrides"]["seed"]: r for r in groups[base_key]}
        shared = [r for r in runs if r["overrides"]["seed"] in base_runs]
        if not shared:
            continue
        for panel in ("iid", "wording", "combo", "long", "full", "reorder"):
            a = [_selected_checkpoint(r)[panel]["flags"] for r in shared]
            b = [_selected_checkpoint(base_runs[r["overrides"]["seed"]])[panel]["flags"] for r in shared]
            mean, lo, hi = _paired_bootstrap(a, b)
            boot_rows.append([key[0], key[1], key[4], panel, str(len(shared)), f"{mean:+.1f}", f"[{lo:+.1f}, {hi:+.1f}]",
                              "yes" if lo > -3 else "no"])
    if boot_rows:
        table += ("\n\narm − english, paired per item at the dev-selected checkpoint, points (90% bootstrap over "
                  "items; non-inferior if the lower bound is above −3):\n\n"
                  + _table(["arm", "corpus", "curriculum", "panel", "seeds", "mean", "90% interval", "non-inferior"], boot_rows))
    steps = sorted({h["step"] for _, runs in curve for r in runs for h in r["history"]})
    for (panel, field, pct), title in ((("iid", "exact", True), "iid exact"),
                                        (("full", "exact", True), "full (held-out) exact"),
                                        (("long", "exact", True), "long (K 13..16) exact")):
        lines = []
        for key, runs in curve:
            by_step = {s: [h[panel][field] for r in runs for h in r["history"]
                           if h["step"] == s and panel in h and h[panel].get(field) is not None] for s in steps}
            lines.append([key[0], key[1], key[4], *(_fmt(by_step[s], pct) if by_step[s] else "–" for s in steps)])
        table += f"\n\n{title} by grid update (early-stopped runs end sooner):\n\n" + _table(
            ["arm", "corpus", "curriculum", *(str(s) for s in steps)], lines)
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
    study3_files = sorted(root.glob("language_study3*.jsonl"))
    study_files = [f for f in sorted(root.glob("language_study*.jsonl")) if f not in study3_files]
    if study_files:
        merged = [r for f in study_files for r in _load(f)]
        sections.append(f"## language_study (merged from {', '.join(f.name for f in study_files)})\n\n{study(merged)}\n")
    if study3_files:
        merged = [r for f in study3_files for r in _load(f)]
        sections.append(f"## language_study3 (merged from {', '.join(f.name for f in study3_files)})\n\n{study3(merged)}\n")
    for name in ("language_benchmark", "language_benchmark_k8"):
        bench = root / f"{name}.json"
        if bench.exists():
            report = {k: v for k, v in json.loads(bench.read_text()).items() if k != "config"}
            sections.append(f"## {name}\n\n{language_benchmark(report)}\n")
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

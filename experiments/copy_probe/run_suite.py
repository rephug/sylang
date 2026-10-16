"""Run a named suite of training configurations in parallel and save results.

    python run_suite.py replica_sweep --workers 4
    python run_suite.py matrix --workers 4

Each run is single-threaded; results go to results/<suite>.jsonl, one JSON
object per run, so an interrupted suite keeps finished runs.
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

HERE = Path(__file__).resolve().parent


def _replica_sweep():
    for lr, batch, seed in itertools.product((1e-4, 3e-4, 1e-3, 3e-3), (1, 8, 32), (0, 1, 2)):
        yield dict(task="replica", steps=192, batch_size=batch, lr=lr, seed=seed, eval_every=48)


def _replica_sweep_v0():
    # First replica variant: distractor "sees" differs in length from "helps".
    for config in _replica_sweep():
        yield config | {"replica_distractor": "sees"}


def _replica_budget():
    # Batch size 1: is the failure the batch size or the number of examples seen?
    # Does upweighting the identifier-initial bytes help at a fixed budget?
    for lr, steps, weight, seed in itertools.product((3e-4, 1e-3), (192, 768, 3072), (1.0, 8.0), (0, 1, 2)):
        yield dict(task="replica", steps=steps, batch_size=1, lr=lr, first_byte_weight=weight, seed=seed,
                   eval_every=max(48, steps // 8))


def _matrix():
    # The discriminating experiment: data design x number of updates.
    for (task, ids, jitter), steps, seed in itertools.product(
        (
            ("replica", "fresh", 0),
            ("copy_none", "pool12", 0),
            ("copy_none", "fresh", 0),
            ("role_select", "pool12", 0),
            ("role_select", "fresh", 0),
            ("role_select", "fresh", 16),
        ),
        (192, 3000),
        (0, 1, 2),
    ):
        yield dict(task=task, ids=ids, jitter=jitter, steps=steps, batch_size=32, lr=1e-3, seed=seed, eval_every=250)


def _batch1_copy():
    # Would the planned COPY/NONE run stall at the English run's batch size and budget?
    for lr, steps, seed in itertools.product((3e-4, 1e-3), (192, 3072), (0, 1, 2)):
        yield dict(task="copy_none", ids="fresh", steps=steps, batch_size=1, lr=lr, seed=seed,
                   eval_every=max(48, steps // 8))


def _balanced12():
    # Replaces the confounded pool12 arm: the project's balanced 12/12 identifiers.
    for (task, steps), seed in itertools.product(
        (("copy_none", 192), ("copy_none", 3000), ("role_select", 3000)), (0, 1, 2)
    ):
        yield dict(task=task, ids="balanced12", jitter=0, steps=steps, batch_size=32, lr=1e-3, seed=seed,
                   eval_every=250)


def _controls():
    for seed in (0, 1, 2):
        yield dict(task="repeat", steps=1500, batch_size=32, lr=1e-3, seed=seed, eval_every=250)


SUITES = {
    "replica_sweep": _replica_sweep,
    "replica_sweep_v0": _replica_sweep_v0,
    "replica_budget": _replica_budget,
    "matrix": _matrix,
    "batch1_copy": _batch1_copy,
    "balanced12": _balanced12,
    "controls": _controls,
}


def _run_one(overrides: dict) -> dict:
    import torch

    torch.set_num_threads(1)
    import train

    args = train.parser().parse_args([])
    for key, value in overrides.items():
        setattr(args, key, value)
    result = train.run(args)
    result["overrides"] = overrides
    return result


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("suite", choices=sorted(SUITES))
    p.add_argument("--workers", type=int, default=os.cpu_count() or 1)
    p.add_argument("--out-dir", default=str(HERE / "results"))
    opts = p.parse_args()
    out = Path(opts.out_dir) / f"{opts.suite}.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if out.exists():
        done = {json.dumps(json.loads(line)["overrides"], sort_keys=True) for line in out.read_text().splitlines()}
    configs = [c for c in SUITES[opts.suite]() if json.dumps(c, sort_keys=True) not in done]
    print(f"{len(configs)} runs to do, {len(done)} already in {out}", flush=True)
    with ProcessPoolExecutor(max_workers=opts.workers) as pool, out.open("a") as handle:
        futures = {pool.submit(_run_one, c): c for c in configs}
        for future in as_completed(futures):
            result = future.result()
            handle.write(json.dumps(result, default=str) + "\n")
            handle.flush()
            summary = {k: v["exact"] for k, v in result["final"].items()}
            print(json.dumps(result["overrides"]), json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()

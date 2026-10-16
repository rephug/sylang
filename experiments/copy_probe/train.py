"""Train the reference model on one task and report copy diagnostics.

Example::

    python train.py --task role_select --ids fresh --steps 3000 --seed 0
    python train.py --task replica --steps 192 --batch-size 8
"""

from __future__ import annotations

import argparse
import json
import random
import time
from collections import Counter
from dataclasses import asdict

import torch

import probes
import tasks
from model import EOS, ModelConfig, TinyByteLM, answer_loss, greedy_generate

# ---------------------------------------------------------------------------
# Data


def make_sampler(args, rng: random.Random):
    """Return ``(sample_batch, eval_sets)``.

    ``sample_batch(n)`` draws training examples. ``eval_sets`` maps a name to a
    list of held-out examples scored by exact byte + EOS match.
    """
    eval_rng = random.Random(10_000 + args.seed)
    if args.task == "replica":
        train, checks, padded_none = tasks.replica_datasets(args.seed, distractor=args.replica_distractor)
        order: list[tasks.Example] = []

        def sample_batch(n):
            nonlocal order
            batch = []
            for _ in range(n):
                if not order:
                    order = list(train)
                    rng.shuffle(order)
                batch.append(order.pop())
            return batch

        seen = [ex for ex in checks if ex.meta.get("seen")]
        unseen = [ex for ex in checks if ex.kind == "select" and not ex.meta.get("seen")]
        empty = [ex for ex in checks if ex.kind == "none"]
        return sample_batch, {
            "select_seen": seen,
            "select_unseen": unseen,
            "none": empty,
            "none_length_matched": padded_none,
        }

    if args.task == "repeat":
        def sample_batch(n):
            return [tasks.repeat_example(rng) for _ in range(n)]

        return sample_batch, {"repeat": [tasks.repeat_example(eval_rng) for _ in range(args.n_eval)]}

    if args.ids == "fresh":
        train_ids, test_ids = tasks.FreshIds("train"), tasks.FreshIds("test")
    elif args.ids == "balanced12":
        train_ids = tasks.balanced_pool("train", args.curriculum)
        test_ids = tasks.balanced_pool("test", args.curriculum)
    else:
        size = int(args.ids.removeprefix("pool"))
        train_ids = tasks.fixed_pool("train", size, seed=args.seed)
        test_ids = tasks.fixed_pool("test", size, seed=args.seed)
    make = tasks.copy_none_example if args.task == "copy_none" else tasks.role_select_example
    positive = "copy" if args.task == "copy_none" else "select"  # also the generator's flag

    def sample_batch(n):
        return [make(rng, train_ids, jitter=args.jitter) for _ in range(n)]

    def eval_set(ids, is_positive):
        return [make(eval_rng, ids, jitter=args.jitter, **{positive: is_positive}) for _ in range(args.n_eval)]

    return sample_batch, {
        f"{positive}_train_ids": eval_set(train_ids, True),
        f"{positive}_heldout_ids": eval_set(test_ids, True),
        "none_heldout_ids": eval_set(test_ids, False),
    }


# ---------------------------------------------------------------------------
# Evaluation


def exact_scores(model, examples: list[tasks.Example]) -> dict:
    prompts = [tasks.prompt_ids(ex.prompt) for ex in examples]
    outputs = greedy_generate(model, prompts, max_new=12)
    correct = 0
    predictions = []
    for ex, out in zip(examples, outputs):
        text = bytes(t for t in out if t < 256).decode("utf-8", "replace")
        ok = out == list(ex.answer.encode()) + [EOS]
        correct += ok
        predictions.append(text if out and out[-1] == EOS else text + "<no-eos>")
    counts = Counter(predictions)
    mode, mode_count = counts.most_common(1)[0]
    return {
        "exact": correct / len(examples),
        "n": len(examples),
        "mode_prediction": mode,
        "mode_fraction": mode_count / len(examples),
        "distinct_predictions": len(counts),
    }


@torch.no_grad()
def teacher_forced_breakdown(model, examples: list[tasks.Example]) -> dict:
    """Accuracy and loss by answer role: first identifier byte vs the rest."""
    batch = tasks.collate(examples)
    logits = model(batch["input_ids"])
    nll = torch.nn.functional.cross_entropy(
        logits.transpose(1, 2), batch["labels"], ignore_index=-100, reduction="none"
    )
    hit = logits.argmax(-1) == batch["labels"]
    stats: dict[str, list[float]] = {}
    for row, ex in enumerate(examples):
        positions = (batch["labels"][row] != -100).nonzero().flatten().tolist()
        for t, role in zip(positions, tasks.answer_roles(ex)):
            stats.setdefault(role, []).append((float(nll[row, t]), float(hit[row, t])))
    return {
        role: {"loss": sum(v[0] for v in values) / len(values), "acc": sum(v[1] for v in values) / len(values)}
        for role, values in stats.items()
    }


def copy_diagnostics(model, examples: list[tasks.Example], rng: random.Random, replacements: list[str]) -> dict:
    """Prompt dependence, role swap and identifier replacement at the answer start."""
    select = [ex for ex in examples if ex.kind in ("select", "copy")]
    if not select:
        return {}
    logits_fn = model
    prompts = [tasks.prompt_ids(ex.prompt) for ex in select]
    dependence = probes.prompt_dependence_probe(logits_fn, prompts).detail
    out = {"prompt_dependence_js_bits": dependence["js_bits"], "answer_start_top": [
        chr(t) if t < 256 else t for t in dependence["mean_top_tokens"]
    ], "answer_start_top_probs": dependence["mean_top_probs"]}

    swap_pairs, replace_pairs = [], []
    for ex, prompt in zip(select, prompts):
        if ex.kind == "select":
            swapped = tasks.swap_roles(ex)
            swap_pairs.append((prompt, tasks.prompt_ids(swapped.prompt), ord(ex.answer[0]), ord(swapped.answer[0])))
        # Rename the first target to an identifier with a different first byte.
        target = ex.targets[0]
        new = target
        while new[0] == target[0] or new in ex.prompt:
            new = rng.choice(replacements)
        renamed = tasks.replace_identifiers(ex, {target: new})
        replace_pairs.append((prompt, tasks.prompt_ids(renamed.prompt), ord(target[0]), ord(new[0])))
    if swap_pairs:
        out["swap"] = probes.counterfactual_probe(logits_fn, swap_pairs)
    out["replace"] = probes.counterfactual_probe(logits_fn, replace_pairs)
    return out


def evaluate(model, eval_sets: dict, rng: random.Random, full: bool = False, replacements=None) -> dict:
    result = {name: exact_scores(model, examples) for name, examples in eval_sets.items()}
    if full:
        for name, examples in eval_sets.items():
            result[name]["teacher_forced"] = teacher_forced_breakdown(model, examples)
            diag = copy_diagnostics(model, examples, rng, replacements or tasks.FreshIds("test").ids)
            if diag:
                result[name]["diagnostics"] = diag
    return result


# ---------------------------------------------------------------------------
# Training


def run(args) -> dict:
    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)
    sample_batch, eval_sets = make_sampler(args, rng)
    cfg = ModelConfig(d_model=args.d_model, n_layers=args.layers, n_heads=args.heads)
    model = TinyByteLM(cfg)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.98), weight_decay=0.01)
    warmup = max(1, min(100, args.steps // 10))
    schedule = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: min(1.0, (step + 1) / warmup))

    history = []
    started = time.perf_counter()
    for step in range(1, args.steps + 1):
        model.train()
        batch = tasks.collate(sample_batch(args.batch_size), first_byte_weight=args.first_byte_weight)
        logits = model(batch["input_ids"])
        loss = answer_loss(logits, batch["labels"], batch["weights"])
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        schedule.step()
        if step % args.eval_every == 0 or step == args.steps:
            scores = evaluate(model, eval_sets, rng)
            entry = {"step": step, "loss": round(loss.item(), 4)} | {k: v["exact"] for k, v in scores.items()}
            history.append(entry)
            if args.verbose:
                print(json.dumps(entry), flush=True)
    seconds = time.perf_counter() - started
    replacements = list(tasks.REPLICA_IDS) if args.task == "replica" else None
    final = evaluate(model, eval_sets, rng, full=True, replacements=replacements)
    return {
        "args": vars(args),
        "model": asdict(cfg),
        "parameters": sum(p.numel() for p in model.parameters()),
        "train_seconds": round(seconds, 1),
        "history": history,
        "final": final,
    }


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--task", choices=("repeat", "copy_none", "role_select", "replica"), default="role_select")
    p.add_argument("--ids", default="fresh", help="'fresh', 'balanced12' or 'poolN' (N train ids and N unseen test ids)")
    p.add_argument("--jitter", type=int, default=0, help="max filler bytes before the facts")
    p.add_argument("--replica-distractor", default="likes", help="distractor relation in the replica task")
    p.add_argument("--curriculum", help="packet copy/source/CURRICULUM.json, for --ids balanced12")
    p.add_argument("--steps", type=int, default=3000)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--first-byte-weight", type=float, default=1.0)
    p.add_argument("--d-model", type=int, default=128)
    p.add_argument("--layers", type=int, default=2)
    p.add_argument("--heads", type=int, default=4)
    p.add_argument("--n-eval", type=int, default=200)
    p.add_argument("--eval-every", type=int, default=250)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--output")
    p.add_argument("--verbose", action="store_true")
    return p


def main() -> None:
    args = parser().parse_args()
    torch.set_num_threads(max(1, torch.get_num_threads()))
    result = run(args)
    text = json.dumps(result, indent=2, default=str)
    if args.output:
        with open(args.output, "w") as handle:
            handle.write(text + "\n")
    print(json.dumps({"final": {k: v["exact"] for k, v in result["final"].items()}, "seconds": result["train_seconds"]}))


if __name__ == "__main__":
    main()

"""Run every mechanical probe on a model and print a pass/fail report.

On the reference model:

    python check_pipeline.py                 # all probes should pass
    python check_pipeline.py --bug detach_kv  # see which probe catches it

To check another codebase, call ``check_pipeline`` with adapters for that
codebase's own model, loss, collate, decoder and checkpoint I/O. The adapters
must call the real training and evaluation code paths, not re-implementations,
or the probes test the adapter instead of the pipeline.
"""

from __future__ import annotations

import argparse
import io
import random
from typing import Callable

import torch

import probes
import tasks
from model import BUGS, PAD, VOCAB_SIZE, ModelConfig, TinyByteLM, answer_loss, greedy_generate


def check_pipeline(
    *,
    model: torch.nn.Module,
    embedding: torch.nn.Module,
    logits_fn: Callable,
    loss_fn: Callable,
    batch: dict,
    vocab_size: int,
    pad_id: int | None,
    generate_fn: Callable[[str], list[int]],
    encode_fn: Callable[[str], list[int]],
    prompts: list[str],
    build_fresh: Callable[[], torch.nn.Module],
    save: Callable,
    load: Callable,
    collate_fn: Callable[[list], dict],
    batch_logits_fn: Callable[[dict], torch.Tensor],
    mixed_length_examples: list,
    optimizer=None,
) -> list[probes.ProbeResult]:
    first = batch["input_ids"][:1]
    supervised = (batch["labels"][0] != -100).nonzero().flatten()
    query = int(supervised.min())  # predicts the first answer byte
    context = first[:, : query + 1]
    results = [
        probes.causality_probe(logits_fn, context, vocab_size),
        probes.visibility_probe(logits_fn, context, query, vocab_size),
        probes.gradient_reach_probe(model, embedding, batch, loss_fn, logits_fn),
    ]
    model.train()
    results.append(probes.parameter_gradient_probe(model, loss_fn(logits_fn(batch["input_ids"]), batch), optimizer))
    model.eval()
    results.append(probes.oracle_label_probe(loss_fn, batch, vocab_size, pad_id))
    results.append(probes.checkpoint_roundtrip_probe(model, build_fresh, save, load, context))
    results.append(probes.decoder_consistency_probe(generate_fn, logits_fn, encode_fn, prompts))
    results.append(probes.padding_invariance_probe(collate_fn, batch_logits_fn, mixed_length_examples))
    return results


def _reference(bug: str | None, seed: int = 0) -> list[probes.ProbeResult]:
    cfg = ModelConfig(d_model=64, n_layers=2, n_heads=4, bug=bug)
    torch.manual_seed(seed)
    model = TinyByteLM(cfg)
    with torch.no_grad():  # make the untrained argmax context sensitive
        model.head.weight.mul_(50)
    rng = random.Random(seed)
    examples = [tasks.role_select_example(rng, tasks.FreshIds("train")) for _ in range(8)]
    examples = [ex for ex in examples if ex.kind == "select"] or examples

    def save(m):
        buffer = io.BytesIO()
        torch.save(m.state_dict(), buffer)
        buffer.seek(0)
        return buffer

    def load(m, buffer):
        m.load_state_dict(torch.load(buffer))

    def build_fresh():
        torch.manual_seed(seed + 1)
        return TinyByteLM(cfg)

    return check_pipeline(
        model=model,
        embedding=model.tok_emb,
        logits_fn=model,
        loss_fn=lambda logits, batch: answer_loss(logits, batch["labels"]),
        batch=tasks.collate(examples),
        vocab_size=VOCAB_SIZE,
        pad_id=PAD,
        generate_fn=lambda prompt: greedy_generate(model, [tasks.prompt_ids(prompt)], max_new=8)[0],
        encode_fn=tasks.prompt_ids,
        prompts=[ex.prompt for ex in examples[:4]],
        build_fresh=build_fresh,
        save=save,
        load=load,
        collate_fn=tasks.collate,
        batch_logits_fn=lambda batch: model(batch["input_ids"]),
        mixed_length_examples=[tasks.copy_none_example(rng, tasks.FreshIds("train"), copy=True), *examples],
        optimizer=torch.optim.AdamW(model.parameters()),
    )


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--bug", choices=[b for b in BUGS if b], default=None)
    args = p.parse_args()
    results = _reference(args.bug)
    for result in results:
        print(result)
    raise SystemExit(0 if all(r.passed for r in results) else 1)


if __name__ == "__main__":
    main()

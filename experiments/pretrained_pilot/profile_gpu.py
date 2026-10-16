#!/usr/bin/env python
"""Stage A profiling: measure this machine, project the cost of stages B and C.

    profile_gpu.py --config C --out profile.json [--price-per-hour P] [--train-steps 30] [--gen-items 64]
               [--upload-cmd CMD] [--device auto|cpu|cuda] [--threads N]

Measures model load seconds, parameter counts, peak memory, training examples/s and tokens/s
(with ``train_lora.train_step``, the trainer's own step), dev generation items/s (with the
trainer's generation routine), checkpoint save seconds and bytes, and, when ``--upload-cmd`` is
given, the seconds of that shell command run with the checkpoint directory appended as its last
argument.  Then every cell of ``configs/plan_stage_b.json`` and ``configs/plan_stage_c.json``
(``{"cells": [{"config", "train_cases", "epochs", "eval_items", "dev_evals", "dev_eval_items"}]}``)
is projected: load seconds + training examples / examples_per_s + generated items /
items_per_s + checkpoint seconds (+ upload seconds); the stage total carries a 1.5x
contingency and is priced at ``--price-per-hour``.  Everything is written to ``profile.json``.
"""
from __future__ import annotations

import argparse
import json
import platform
import shlex
import subprocess
import tempfile
from pathlib import Path

import torch

import common
import train_lora as tl

CONTINGENCY = 1.5
PLANS = {'stage_b': 'plan_stage_b.json', 'stage_c': 'plan_stage_c.json'}


def sync(device):
    if device == 'cuda':
        torch.cuda.synchronize()


def measure(args):
    if args.train_steps < 1 or args.gen_items < 1:
        raise SystemExit('--train-steps and --gen-items must be at least 1')
    config = common.load_config(args.config)
    device = common.resolve_device(args.device)
    if device == 'cpu':
        torch.set_num_threads(args.threads)
    if device == 'cuda':
        torch.cuda.reset_peak_memory_stats()
    total = common.Stopwatch()

    watch = common.Stopwatch()
    tok = common.load_tokenizer()
    common.check_template(tok)
    tokenizer_seconds = watch.seconds()
    legend = common.load_legends(config['data_version'])[config['answer_format']][config['arm']]
    model_path = common.resolve_model_path(config)
    watch = common.Stopwatch()
    model, info = tl.build_model(config, device, model_path)
    sync(device)
    model_seconds = watch.seconds()
    autocast_dtype = info['autocast_dtype']
    print(f'load: tokenizer {tokenizer_seconds:.1f}s, model {model_seconds:.1f}s, {info["model_params"]:,} params, '
          f'{info["trainable_params"]:,} trainable, {device} {info["dtype"]}', flush=True)

    watch = common.Stopwatch()
    rows = common.training_examples(config)
    items, dropped = tl.tokenise_examples(rows, tok, legend, config)
    tokenise_seconds = watch.seconds()
    if not items:
        raise SystemExit(f'no training examples: read {len(rows)}, dropped {dropped} longer than max_length')

    # Training throughput: one untimed warm-up step, then --train-steps timed steps of the config's
    # batch (batch_size x grad_accum examples), cycling through the seed's shuffled stream.
    bs, ga, steps = config['batch_size'], config['grad_accum'], args.train_steps
    perm = tl.epoch_permutation(len(items), config['seed'], 1)
    order = [perm[i % len(perm)] for i in range((steps + 1) * ga * bs)]
    optimizer, scheduler = tl.build_optimizer(model, config, steps + 1)

    def group(k):
        base = k * ga * bs
        return [tl.collate([items[i] for i in order[base + m * bs: base + (m + 1) * bs]], tok.pad_token_id)
                for m in range(ga)]

    tl.train_step(model, optimizer, scheduler, group(0), device, autocast_dtype)
    sync(device)
    watch = common.Stopwatch()
    examples = tokens = loss_tokens = 0
    losses = []
    for k in range(1, steps + 1):
        r = tl.train_step(model, optimizer, scheduler, group(k), device, autocast_dtype)
        examples += r['examples']
        tokens += r['tokens']
        loss_tokens += r['loss_tokens']
        losses.append(r['loss'])
    sync(device)
    train_seconds = watch.seconds()
    training = {'steps': steps, 'batch_size': bs, 'grad_accum': ga, 'examples': examples, 'tokens': tokens,
                'completion_tokens': loss_tokens, 'seconds': train_seconds,
                'examples_per_s': examples / train_seconds, 'tokens_per_s': tokens / train_seconds,
                'seconds_per_step': train_seconds / steps, 'tokens_per_example': tokens / examples,
                'first_loss': losses[0], 'last_loss': losses[-1],
                'train_examples_available': len(items), 'dropped_too_long': dropped}
    print(f'training: {steps} steps in {train_seconds:.1f}s = {training["examples_per_s"]:.2f} examples/s, '
          f'{training["tokens_per_s"]:.0f} tokens/s', flush=True)

    # Generation throughput on dev examples, balanced over tasks, with the trainer's routine.
    dev = tl.select_dev_examples(config, args.gen_items)
    sync(device)
    watch = common.Stopwatch()
    outputs = tl.dev_generate(model, tok, legend, dev, config, device, autocast_dtype)
    sync(device)
    gen_seconds = watch.seconds()
    out_tokens = [len(tok(o, add_special_tokens=False)['input_ids']) for o in outputs]
    by_task = {t: sum(1 for ex in dev if ex['task'] == t) for t in config['tasks']}
    generation = {'items': len(dev), 'by_task': by_task, 'seconds': gen_seconds,
                  'items_per_s': len(dev) / gen_seconds if gen_seconds else None,
                  'batch_size': bs, 'max_new_tokens': config['max_new_tokens'],
                  'output_tokens_mean': sum(out_tokens) / len(out_tokens) if out_tokens else None}
    print(f'generation: {len(dev)} items in {gen_seconds:.1f}s = {generation["items_per_s"]:.2f} items/s '
          f'(mean {generation["output_tokens_mean"]:.1f} output tokens)', flush=True)

    # Checkpoint save (adapter + trainer state) and the optional upload of that directory.
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    upload = None
    with tempfile.TemporaryDirectory(prefix='.profile_ckpt_', dir=out_path.parent) as tmp:
        ckpt = Path(tmp) / 'epoch1'
        state = {'step': steps + 1, 'epoch': 1, 'epoch_complete': True, 'batches_done': (steps + 1) * ga,
                 'profile': True, 'config': config, 'optimizer': optimizer.state_dict(),
                 'scheduler': scheduler.state_dict(), 'rng': tl.rng_state()}
        watch = common.Stopwatch()
        ckpt_bytes = tl.save_checkpoint(model, ckpt, state)
        ckpt_seconds = watch.seconds()
        checkpoint = {'seconds': ckpt_seconds, 'bytes': ckpt_bytes,
                      'adapter_bytes': ckpt_bytes - (ckpt / tl.STATE_FILE).stat().st_size}
        print(f'checkpoint: {ckpt_bytes:,} bytes in {ckpt_seconds:.2f}s', flush=True)
        if args.upload_cmd:
            cmd = f'{args.upload_cmd} {shlex.quote(str(ckpt))}'
            watch = common.Stopwatch()
            proc = subprocess.run(cmd, shell=True, capture_output=True, text=True)
            upload = {'command': cmd, 'seconds': watch.seconds(), 'returncode': proc.returncode,
                      'stdout_tail': proc.stdout[-500:], 'stderr_tail': proc.stderr[-500:]}
            print(f'upload: {upload["seconds"]:.1f}s, exit {proc.returncode}', flush=True)

    return {'name': config['name'], 'config': config, 'git_commit': common.git_commit(),
            'measured_at': common.now_iso(), 'host': platform.node(), 'device': device,
            'gpu_name': torch.cuda.get_device_name(0) if device == 'cuda' else None,
            'threads': torch.get_num_threads() if device == 'cpu' else None,
            'versions': {'torch': torch.__version__, 'transformers': __import__('transformers').__version__,
                         'peft': __import__('peft').__version__, 'python': platform.python_version()},
            'dtype': info['dtype'], 'model_path': info['model_path'], 'model_params': info['model_params'],
            'trainable_params': info['trainable_params'],
            'load': {'tokenizer_seconds': tokenizer_seconds, 'model_seconds': model_seconds,
                     'seconds': tokenizer_seconds + model_seconds, 'tokenise_seconds': tokenise_seconds},
            'training': training, 'generation': generation, 'checkpoint': checkpoint, 'upload': upload,
            'peak_memory_bytes': tl.peak_memory_bytes(device), 'seconds': total.seconds()}


def project(measured, price, plan_dir=None):
    """Projected seconds and cost per cell and per stage from the measured rates.

    Returns ``(projection, notes)``; a stage whose plan file is missing projects to None, and when
    both are missing the projection itself is None."""
    plan_dir = Path(plan_dir or common.CONFIG_DIR)
    load_s = measured['load']['seconds']
    ex_per_s = measured['training']['examples_per_s']
    items_per_s = measured['generation']['items_per_s']
    ckpt_s = measured['checkpoint']['seconds']
    upload_s = measured['upload']['seconds'] if measured.get('upload') else 0.0
    notes, stages = [], {}
    for stage, fname in PLANS.items():
        path = plan_dir / fname
        if not path.exists():
            stages[stage] = None
            notes.append(f'{stage}: {path} not found, no projection')
            continue
        with open(path, encoding='utf-8') as f:
            plan = json.load(f)
        cells = []
        for cell in plan['cells']:
            n_tasks, timeout_min, epochs = 2, None, cell.get('epochs')
            try:
                cfg = common.load_config(common.HERE / cell['config'])
                n_tasks, timeout_min = len(cfg['tasks']), cfg['timeout_minutes']
                epochs = epochs if epochs is not None else cfg['epochs']
            except Exception as exc:  # noqa: BLE001 - a plan may name configs not written yet
                notes.append(f'{stage}: {cell["config"]}: {type(exc).__name__}: {exc}; assuming 2 examples per case')
                epochs = epochs if epochs is not None else common.DEFAULT_CONFIG['epochs']
            train_examples = n_tasks * cell['train_cases'] * epochs
            gen_items = cell['dev_evals'] * cell['dev_eval_items'] + cell['eval_items']
            parts = {'load_s': load_s, 'train_s': train_examples / ex_per_s, 'generation_s': gen_items / items_per_s,
                     'checkpoint_s': ckpt_s * epochs, 'upload_s': upload_s}
            seconds = sum(parts.values())
            cells.append({'config': cell['config'], 'train_cases': cell['train_cases'], 'epochs': epochs,
                          'examples_per_case': n_tasks, 'train_examples': train_examples,
                          'generation_items': gen_items, **parts, 'seconds': seconds,
                          'cost': seconds / 3600 * price, 'timeout_minutes': timeout_min,
                          'exceeds_timeout': (seconds > timeout_min * 60) if timeout_min else None})
        raw = sum(c['seconds'] for c in cells)
        stages[stage] = {'plan': str(path), 'n_cells': len(cells), 'cells': cells, 'seconds': raw,
                         'contingency': CONTINGENCY, 'seconds_with_contingency': raw * CONTINGENCY,
                         'hours_with_contingency': raw * CONTINGENCY / 3600,
                         'cost': raw * CONTINGENCY / 3600 * price, 'price_per_hour': price}
    if all(v is None for v in stages.values()):
        return None, notes
    return stages, notes


def parse_args(argv=None):
    p = argparse.ArgumentParser(description='Stage A profiling (see interfaces.md, Profiling).')
    p.add_argument('--config', required=True)
    p.add_argument('--out', required=True, help='profile.json')
    p.add_argument('--price-per-hour', type=float, default=0.0)
    p.add_argument('--train-steps', type=int, default=30)
    p.add_argument('--gen-items', type=int, default=64)
    p.add_argument('--upload-cmd', default=None, help='shell command; the checkpoint directory is appended')
    p.add_argument('--device', default='auto', choices=['auto', 'cpu', 'cuda'])
    p.add_argument('--threads', type=int, default=2, help='torch threads on CPU')
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    measured = measure(args)
    projection, notes = project(measured, args.price_per_hour)
    profile = {**measured, 'price_per_hour': args.price_per_hour, 'projection': projection, 'notes': notes}
    common.write_json(args.out, profile)
    for note in notes:
        print('note:', note)
    if projection:
        for stage, proj in projection.items():
            if proj:
                print(f'{stage}: {proj["n_cells"]} cells, {proj["seconds"] / 3600:.2f} h raw, '
                      f'{proj["hours_with_contingency"]:.2f} h with contingency, ${proj["cost"]:.2f} '
                      f'at ${args.price_per_hour}/h')
    print(f'wrote {args.out} (peak memory {profile["peak_memory_bytes"] / 2**20:.0f} MiB)')


if __name__ == '__main__':
    main()

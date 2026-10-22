#!/usr/bin/env python
"""Stage A profiling: measure this machine, project the cost of stages B and C.

    profile_gpu.py --config C --out profile.json [--price-per-hour P] [--train-steps 30] [--gen-items 64]
               [--upload-cmd CMD] [--device auto|cpu|cuda] [--threads N] [--plan NAME=FILE ...]

Measures model load seconds, parameter counts, peak memory, training examples/s and tokens/s
(with ``train_lora.train_step``, the trainer's own step), dev generation items/s (with the
trainer's generation routine), checkpoint save seconds and bytes, and, when ``--upload-cmd`` is
given, the seconds of that shell command run with the checkpoint directory appended as its last
argument; of the command ``profile.json`` records only its program name, seconds and exit status,
never the command line or its output (either may carry credentials; a failing command's stderr
tail goes to the console only).  Stage A profiles the ``json`` arm (``configs/stage_a_profile.json``),
the longest rendering, so the single-rate projection over-estimates the other arms.

Every plan file (``configs/plan_stage_b.json`` and ``configs/plan_stage_c.json``, or the ``--plan``
overrides) is read and checked before anything is measured, so a malformed plan aborts before any
GPU time; the measured profile is written before the projection runs, and a projection error is
recorded in ``notes`` (``projection failed: ...; measured values kept``) with the measured values
kept and a non-zero exit.  Every cell
(``{"cells": [{"config", "train_cases", "epochs", "eval_items", "dev_evals", "dev_eval_items"}]}``)
is projected in two phases, each with its own model load because ``train_lora.py`` and
``evaluate.py`` each load the model: the training phase (load + training examples / examples_per_s
+ dev generation items / items_per_s + checkpoint seconds x epochs: what the ``train_lora.py``
timeout covers) and the evaluation phase (load + test items / items_per_s: what the ``evaluate.py``
timeout covers), plus the upload seconds.  A cell whose longer phase exceeds its config's
``timeout_minutes`` is flagged (``exceeds_timeout``, a ``!!!`` line), as is one whose first epoch
alone would (``epoch_exceeds_timeout``: no checkpoint would ever be written).  The stage total
carries a 1.5x contingency and is priced at ``--price-per-hour``.  Everything is written to
``profile.json``.
"""
from __future__ import annotations

import argparse
import json
import platform
import shlex
import subprocess
import tempfile
import traceback
from pathlib import Path

import sys

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:  # python -I does not add the script's directory
    sys.path.insert(0, str(HERE))

import torch

import common
import train_lora as tl

CONTINGENCY = 1.5
PLANS = {'stage_b': 'plan_stage_b.json', 'stage_c': 'plan_stage_c.json'}
PLAN_CELL_KEYS = ('config', 'train_cases', 'dev_evals', 'dev_eval_items', 'eval_items')  # every cell needs these


def sync(device):
    if device == 'cuda':
        torch.cuda.synchronize()


def upload_program_name(upload_cmd):
    """The first word of ``--upload-cmd``: all that profile.json records of it (the command line and
    its output may carry credentials).  None without a command; SystemExit on an unparsable one."""
    if not upload_cmd:
        return None
    try:
        words = shlex.split(upload_cmd)
    except ValueError as exc:
        raise SystemExit(f'--upload-cmd: {exc}')
    if not words:
        raise SystemExit('--upload-cmd is empty')
    return words[0]


def parse_plan_args(plan_args):
    """``--plan NAME=FILE`` items to ``{name: file}``; None without overrides (the default ``PLANS``)."""
    if not plan_args:
        return None
    plans = {}
    for item in plan_args:
        name, sep, fname = item.partition('=')
        if not sep or not name or not fname:
            raise SystemExit(f'--plan {item!r}: expected NAME=FILE')
        plans[name] = fname
    return plans


def load_plans(plans=None, plan_dir=None):
    """Read and check every plan file before anything is measured, so a malformed plan aborts before
    any GPU time.  Returns ``{stage: (path, plan)}``; a missing file maps to ``(path, None)`` and
    ``project`` notes it.  Every cell must carry ``PLAN_CELL_KEYS``, the four counts as numbers."""
    plan_dir = Path(plan_dir or common.CONFIG_DIR)
    loaded = {}
    for stage, fname in (plans or PLANS).items():
        path = plan_dir / fname
        if not path.exists():
            loaded[stage] = (path, None)
            continue
        try:
            with open(path, encoding='utf-8') as f:
                plan = json.load(f)
        except ValueError as exc:
            raise SystemExit(f'{path}: not valid JSON: {exc}')
        cells = plan.get('cells') if isinstance(plan, dict) else None
        if not isinstance(cells, list):
            raise SystemExit(f'{path}: no "cells" list')
        for i, cell in enumerate(cells):
            if not isinstance(cell, dict):
                raise SystemExit(f'{path}: cell {i} is not an object')
            label = cell.get('name') or cell.get('config') or '?'
            missing = [k for k in PLAN_CELL_KEYS if k not in cell]
            if missing:
                raise SystemExit(f'{path}: cell {i} ({label}) lacks {missing}')
            not_numbers = [k for k in PLAN_CELL_KEYS[1:]
                           if isinstance(cell[k], bool) or not isinstance(cell[k], (int, float))]
            if not_numbers:
                raise SystemExit(f'{path}: cell {i} ({label}): {not_numbers} must be numbers')
        loaded[stage] = (path, plan)
    return loaded


def measure(args):
    if args.train_steps < 1 or args.gen_items < 1:
        raise SystemExit('--train-steps and --gen-items must be at least 1')
    upload_program = upload_program_name(args.upload_cmd)  # checked before any GPU time
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
        if upload_program is not None:
            cmd = f'{args.upload_cmd} {shlex.quote(str(ckpt))}'
            watch = common.Stopwatch()
            proc = subprocess.run(cmd, shell=True, capture_output=True, text=True)
            # Only the program name, the seconds and the exit status are recorded: the command line
            # and its output may carry credentials.
            upload = {'program': upload_program, 'seconds': watch.seconds(), 'returncode': proc.returncode}
            print(f'upload ({upload_program}): {upload["seconds"]:.1f}s, exit {proc.returncode}', flush=True)
            if proc.returncode != 0:  # console only, never profile.json
                print(f'upload stderr (last 500 chars):\n{proc.stderr[-500:]}', file=sys.stderr, flush=True)

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


def project(measured, price, plans):
    """Projected seconds and cost per cell and per stage from the measured rates, for the plans
    ``load_plans`` returned.

    Each cell is two processes, each loading the model: ``train_lora.py`` (``train_phase_s`` = load +
    training + dev generation + checkpoints, under the config's ``timeout_minutes``) and
    ``evaluate.py`` (``eval_phase_s`` = load + test generation, under the same timeout); ``epoch_s``
    is the time to the first checkpoint.  Returns ``(projection, notes)``; a stage whose plan file
    is missing projects to None, and when every plan is missing the projection itself is None."""
    load_s = measured['load']['seconds']
    ex_per_s = measured['training']['examples_per_s']
    items_per_s = measured['generation']['items_per_s']
    ckpt_s = measured['checkpoint']['seconds']
    upload_s = measured['upload']['seconds'] if measured.get('upload') else 0.0
    notes, stages = [], {}
    for stage, (path, plan) in plans.items():
        if plan is None:
            stages[stage] = None
            notes.append(f'{stage}: {path} not found, no projection')
            continue
        cells = []
        for cell in plan['cells']:
            n_tasks, timeout_min, epochs = 2, cell.get('timeout_minutes'), cell.get('epochs')
            try:
                cfg = common.load_config(common.HERE / cell['config'])
                n_tasks, timeout_min = len(cfg['tasks']), cfg['timeout_minutes']
                epochs = epochs if epochs is not None else cfg['epochs']
            except Exception as exc:  # noqa: BLE001 - a plan may name configs not written yet
                notes.append(f'{stage}: {cell["config"]}: {type(exc).__name__}: {exc}; assuming 2 examples per case')
                epochs = epochs if epochs is not None else common.DEFAULT_CONFIG['epochs']
            train_examples = n_tasks * cell['train_cases'] * epochs
            dev_gen_items = cell['dev_evals'] * cell['dev_eval_items']
            train_s = train_examples / ex_per_s
            dev_gen_s = dev_gen_items / items_per_s
            test_gen_s = cell['eval_items'] / items_per_s
            train_phase_s = load_s + train_s + dev_gen_s + ckpt_s * epochs   # train_lora.py, under its timeout
            eval_phase_s = load_s + test_gen_s                                # evaluate.py, under its timeout
            epoch_s = load_s + (train_s + dev_gen_s) / max(1, epochs) + ckpt_s  # time to the first checkpoint
            seconds = train_phase_s + eval_phase_s + upload_s                 # the model load counted twice
            limit_s = timeout_min * 60 if timeout_min else None
            cells.append({'config': cell['config'], 'train_cases': cell['train_cases'], 'epochs': epochs,
                          'examples_per_case': n_tasks, 'train_examples': train_examples,
                          'dev_gen_items': dev_gen_items, 'test_gen_items': cell['eval_items'],
                          'load_s': load_s, 'train_s': train_s, 'dev_gen_s': dev_gen_s, 'test_gen_s': test_gen_s,
                          'checkpoint_s': ckpt_s * epochs, 'upload_s': upload_s,
                          'train_phase_s': train_phase_s, 'eval_phase_s': eval_phase_s, 'epoch_s': epoch_s,
                          'seconds': seconds, 'cost': seconds / 3600 * price, 'timeout_minutes': timeout_min,
                          'exceeds_timeout': (max(train_phase_s, eval_phase_s) > limit_s) if limit_s else None,
                          'epoch_exceeds_timeout': (epoch_s > limit_s) if limit_s else None})
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
    p.add_argument('--plan', action='append', default=None, metavar='NAME=FILE',
                   help='project this plan file instead of the stage B and C plans (repeatable), e.g. '
                        'stage_d_1p7b=plan_stage_d_1p7b.json when profiling the 1.7B weights')
    return p.parse_args(argv)


def timeout_warnings(projection):
    """One ``!!!`` line per cell whose projected training or evaluation phase (or first epoch alone)
    exceeds its config's ``timeout_minutes``; printed and kept in profile.json's ``notes``."""
    lines = []
    for stage, proj in (projection or {}).items():
        for c in (proj or {}).get('cells', ()):
            if not (c['exceeds_timeout'] or c['epoch_exceeds_timeout']):
                continue
            first = (f', first epoch alone {c["epoch_s"] / 60:.0f} min so no checkpoint would be written'
                     if c['epoch_exceeds_timeout'] else '')
            lines.append(f'!!! {stage} {c["config"]}: projected training {c["train_phase_s"] / 60:.0f} min, '
                         f'evaluation {c["eval_phase_s"] / 60:.0f} min vs timeout_minutes {c["timeout_minutes"]}'
                         f'{first}: raise timeout_minutes or shrink the cell before starting the stage')
    return lines


def main(argv=None):
    args = parse_args(argv)
    plans = load_plans(parse_plan_args(args.plan))  # before measure(): a malformed plan costs no GPU time
    measured = measure(args)
    profile = {**measured, 'price_per_hour': args.price_per_hour, 'projection': None, 'notes': []}
    common.write_json(args.out, profile)  # the measurement is on disk before the projection can fail
    failure = None
    try:
        projection, notes = project(measured, args.price_per_hour, plans)
    except Exception as exc:  # noqa: BLE001 - whatever broke here, the measured values are kept
        traceback.print_exc()
        failure = f'projection failed: {type(exc).__name__}: {exc}; measured values kept'
        projection, notes = None, [failure]
    notes = notes + timeout_warnings(projection)
    profile['projection'], profile['notes'] = projection, notes
    common.write_json(args.out, profile)
    for note in notes:
        print(note if note.startswith('!!!') else f'note: {note}', flush=True)
    if projection:
        for stage, proj in projection.items():
            if proj:
                print(f'{stage}: {proj["n_cells"]} cells, {proj["seconds"] / 3600:.2f} h raw, '
                      f'{proj["hours_with_contingency"]:.2f} h with contingency, ${proj["cost"]:.2f} '
                      f'at ${args.price_per_hour}/h')
    print(f'wrote {args.out} (peak memory {profile["peak_memory_bytes"] / 2**20:.0f} MiB)')
    if failure:
        raise SystemExit(f'profile_gpu.py: {failure} in {args.out}')


if __name__ == '__main__':
    main()

#!/usr/bin/env python
"""LoRA fine-tuning for the pretrained-model pilot.

    train_lora.py --config configs/<cell>.json --out runs/<name> [--resume] [--device auto|cpu|cuda]
                  [--max-steps N] [--log-every N] [--threads N]

A plain PyTorch loop (no HF Trainer) over Transformers + PEFT.  The recipe is the config
(``common.DEFAULT_CONFIG``) and nothing else: prompt tokens are masked out of the loss, batches
are right-padded dynamically, examples longer than ``max_length`` are dropped and counted, AdamW
with linear warmup then cosine decay to ``lr_floor`` x lr, gradient accumulation, gradient
clipping at ``MAX_GRAD_NORM``, bf16 base weights with float32 LoRA parameters under autocast on
CUDA (float32 everywhere on CPU).  At every epoch end, and when ``max_steps`` stops the run
early, it records the mean training loss, scores ``dev_eval_examples`` dev examples by greedy
generation (``grader.grade``), writes ``checkpoints/epoch<N>/`` (adapter + ``trainer_state.pt``),
refreshes ``best/`` (highest dev accuracy, ties to the later epoch) and rewrites ``run.json``;
formats are in ``interfaces.md``.  Dev outputs are decoded with ``common.decode_output`` (the text
before the first <|im_end|>, special tokens skipped), the rule ``evaluate.py`` grades by, so dev
selection sees exactly the text the evaluator sees.  ``--resume`` continues from the last
checkpoint, including one written by an early ``max_steps`` stop in the middle of an epoch; it
re-derives ``best/`` from the checkpoint the saved state names (a kill may have lost the copy) and
drops the ``train_log.jsonl`` lines past the checkpoint's step (the resumed run writes them again),
so the log never holds a step twice.  ``profile_gpu.py`` imports the building blocks below instead
of duplicating them.
"""
from __future__ import annotations

import argparse
import contextlib
import json
import math
import os
import random
import shutil
import traceback
from pathlib import Path

import sys

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:  # python -I does not add the script's directory
    sys.path.insert(0, str(HERE))


import torch

import common

MAX_GRAD_NORM = 1.0          # fixed part of the recipe (not a config key)
STATE_FILE = 'trainer_state.pt'
EPOCH_SEED_STRIDE = 1_000_003


# ---------------------------------------------------------------------------
# Data

def tokenise_examples(rows, tok, legend, config):
    """Prompt + completion tokens per example with the prompt masked out of the labels.

    Returns ``(items, n_dropped)``; items longer than ``max_length`` are dropped and counted."""
    fmt, max_length = config['answer_format'], config['max_length']
    items, dropped = [], 0
    for ex in rows:
        prompt_ids = tok(common.prompt_text(legend, common.user_text(ex, fmt)), add_special_tokens=False)['input_ids']
        completion_ids = tok(common.completion_text(common.answer_text(ex, fmt)), add_special_tokens=False)['input_ids']
        if len(prompt_ids) + len(completion_ids) > max_length:
            dropped += 1
            continue
        items.append({'id': ex['id'], 'task': ex['task'], 'input_ids': prompt_ids + completion_ids,
                      'labels': [-100] * len(prompt_ids) + completion_ids})
    return items, dropped


def collate(items, pad_id):
    """Right-pad to the longest item of the batch; padding gets attention 0 and label -100."""
    width = max(len(it['input_ids']) for it in items)
    input_ids = torch.full((len(items), width), pad_id, dtype=torch.long)
    attention = torch.zeros((len(items), width), dtype=torch.long)
    labels = torch.full((len(items), width), -100, dtype=torch.long)
    for i, it in enumerate(items):
        n = len(it['input_ids'])
        input_ids[i, :n] = torch.tensor(it['input_ids'], dtype=torch.long)
        attention[i, :n] = 1
        labels[i, :n] = torch.tensor(it['labels'], dtype=torch.long)
    return {'input_ids': input_ids, 'attention_mask': attention, 'labels': labels}


def epoch_permutation(n, seed, epoch):
    """The shuffle of epoch ``epoch`` (1-based), a pure function of the seed so a resumed run replays it."""
    gen = torch.Generator().manual_seed(seed * EPOCH_SEED_STRIDE + epoch)
    return torch.randperm(n, generator=gen).tolist()


def select_dev_examples(config, n):
    """``n`` dev examples balanced over the config's tasks: the tasks interleaved in case order."""
    rows = common.read_jsonl(common.examples_path(config['data_version'], 'dev', config['arm']))
    by_task = {t: [r for r in rows if r['task'] == t] for t in config['tasks']}
    out = []
    for i in range(max(len(v) for v in by_task.values())):
        for t in config['tasks']:
            if i < len(by_task[t]):
                out.append(by_task[t][i])
    return out[:n]


# ---------------------------------------------------------------------------
# Model, optimizer, one step

def autocast_context(autocast_dtype):
    if autocast_dtype is None:
        return contextlib.nullcontext()
    return torch.autocast(device_type='cuda', dtype=autocast_dtype)


def build_model(config, device, model_path=None, adapter_dir=None):
    """The base model (bf16 on CUDA, float32 on CPU) wrapped with LoRA; LoRA parameters are float32.

    ``adapter_dir`` loads a saved adapter for resumption instead of initialising a new one.
    Returns ``(model, info)`` with the parameter counts, dtype and the autocast dtype (None on CPU)."""
    from peft import LoraConfig, PeftModel, get_peft_model
    model_path = model_path or common.resolve_model_path(config)
    dtype = common.resolve_dtype(config['dtype'], device)
    base = common.load_model(model_path, device, dtype, trainable=True)
    model_params = sum(p.numel() for p in base.parameters())
    for p in base.parameters():
        p.requires_grad_(False)
    torch.manual_seed(config['seed'])  # LoRA A initialisation
    if adapter_dir is not None:
        model = PeftModel.from_pretrained(base, str(adapter_dir), is_trainable=True)
    else:
        lora = config['lora']
        model = get_peft_model(base, LoraConfig(task_type='CAUSAL_LM', r=lora['r'], lora_alpha=lora['alpha'],
                                                lora_dropout=lora['dropout'], bias='none',
                                                target_modules=list(lora['target_modules'])))
    trainable = 0
    for p in model.parameters():
        if p.requires_grad:
            if p.dtype != torch.float32:
                p.data = p.data.float()
            trainable += p.numel()
    model.train()
    autocast_dtype = torch.bfloat16 if device == 'cuda' and dtype == torch.bfloat16 else None
    info = {'model_path': str(model_path), 'dtype': str(dtype).replace('torch.', ''),
            'model_params': model_params, 'trainable_params': trainable, 'autocast_dtype': autocast_dtype}
    return model, info


def lr_lambda(total_steps, warmup_ratio, lr_floor):
    """Linear warmup over ``warmup_ratio`` of the steps, then cosine decay to ``lr_floor`` of the peak."""
    warmup = int(math.ceil(warmup_ratio * total_steps))

    def factor(step):
        if step < warmup:
            return (step + 1) / warmup
        progress = min(1.0, (step - warmup) / max(1, total_steps - warmup))
        return lr_floor + (1.0 - lr_floor) * 0.5 * (1.0 + math.cos(math.pi * progress))
    return factor


def build_optimizer(model, config, total_steps):
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=config['lr'], weight_decay=config['weight_decay'])
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lr_lambda(total_steps, config['warmup_ratio'], config['lr_floor']))
    return optimizer, scheduler


def train_step(model, optimizer, scheduler, batches, device, autocast_dtype):
    """One optimizer step over ``batches`` (the gradient-accumulation group of collated micro-batches).

    Returns the token-weighted mean loss, the number of loss (completion) tokens, the input tokens
    and examples consumed, the gradient norm and the learning rate that was applied."""
    model.train()
    lr = optimizer.param_groups[0]['lr']
    loss_sum = 0.0  # token-weighted, accumulated on the device; read back once per step
    loss_tokens = tokens = examples = 0
    for batch in batches:
        n_target = int((batch['labels'][:, 1:] != -100).sum())  # counted on the CPU tensors
        tokens += int(batch['attention_mask'].sum())
        examples += int(batch['input_ids'].shape[0])
        batch = {k: v.to(device) for k, v in batch.items()}
        with autocast_context(autocast_dtype):
            out = model(input_ids=batch['input_ids'], attention_mask=batch['attention_mask'],
                        labels=batch['labels'], use_cache=False)
        (out.loss / len(batches)).backward()
        loss_sum = loss_sum + out.loss.detach().double() * n_target
        loss_tokens += n_target
    grad_norm = torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], MAX_GRAD_NORM)
    optimizer.step()
    scheduler.step()
    optimizer.zero_grad(set_to_none=True)
    return {'loss': float(loss_sum) / max(1, loss_tokens), 'loss_tokens': loss_tokens, 'tokens': tokens,
            'examples': examples, 'grad_norm': float(grad_norm), 'lr': lr}


# ---------------------------------------------------------------------------
# Generation and dev evaluation

@torch.no_grad()
def generate_outputs(model, tok, prompts, max_new_tokens, device, autocast_dtype, batch_size):
    """Greedy, batched, left-padded generation; each output is ``common.decode_output`` of the new
    tokens: the text before the first <|im_end|>, special tokens skipped, exactly as evaluate.py grades it."""
    im_end = tok.convert_tokens_to_ids(common.IM_END)
    pad_id = tok.pad_token_id
    encoded = [tok(p, add_special_tokens=False)['input_ids'] for p in prompts]
    order = sorted(range(len(prompts)), key=lambda i: len(encoded[i]))
    outputs = [None] * len(prompts)
    was_training = model.training
    model.eval()
    for start in range(0, len(order), batch_size):
        idx = order[start:start + batch_size]
        width = max(len(encoded[i]) for i in idx)
        input_ids = torch.full((len(idx), width), pad_id, dtype=torch.long)
        attention = torch.zeros((len(idx), width), dtype=torch.long)
        for row, i in enumerate(idx):
            ids = encoded[i]
            input_ids[row, width - len(ids):] = torch.tensor(ids, dtype=torch.long)
            attention[row, width - len(ids):] = 1
        with autocast_context(autocast_dtype):
            gen = model.generate(input_ids=input_ids.to(device), attention_mask=attention.to(device),
                                 max_new_tokens=max_new_tokens, do_sample=False, num_beams=1,
                                 eos_token_id=im_end, pad_token_id=pad_id, use_cache=True,
                                 temperature=None, top_p=None, top_k=None)
        for row, i in enumerate(idx):
            outputs[i] = common.decode_output(tok, gen[row, width:].tolist())
    if was_training:
        model.train()
    return outputs


def dev_generate(model, tok, legend, examples, config, device, autocast_dtype, batch_size=None):
    """Outputs for ``examples`` (aligned), generated per task with the task's ``max_new_tokens``."""
    fmt = config['answer_format']
    batch_size = batch_size or config['batch_size']
    outputs = [None] * len(examples)
    for task in config['tasks']:
        idx = [i for i, ex in enumerate(examples) if ex['task'] == task]
        if not idx:
            continue
        prompts = [common.prompt_text(legend, common.user_text(examples[i], fmt)) for i in idx]
        outs = generate_outputs(model, tok, prompts, config['max_new_tokens'][task], device, autocast_dtype, batch_size)
        for i, out in zip(idx, outs):
            outputs[i] = out
    return outputs


def dev_evaluate(model, tok, legend, examples, config, device, autocast_dtype, epoch):
    """The ``dev`` entry of run.json for one epoch: greedy generation graded by ``grader.grade``."""
    from grader import grade
    watch = common.Stopwatch()
    fmt = config['answer_format']
    outputs = dev_generate(model, tok, legend, examples, config, device, autocast_dtype)
    counts = {t: {'n': 0, 'correct': 0, 'invalid': 0} for t in config['tasks']}
    for ex, out in zip(examples, outputs):
        g = grade(ex['task'], fmt, common.answer_text(ex, fmt), out)
        c = counts[ex['task']]
        c['n'] += 1
        c['correct'] += int(bool(g['correct']))
        c['invalid'] += int(bool(g['invalid']))
    n = sum(c['n'] for c in counts.values())
    correct = sum(c['correct'] for c in counts.values())
    invalid = sum(c['invalid'] for c in counts.values())
    return {'epoch': epoch, 'accuracy': correct / n if n else 0.0, 'invalid_rate': invalid / n if n else 0.0,
            'by_task': {t: {'n': c['n'], 'accuracy': c['correct'] / c['n'] if c['n'] else 0.0,
                            'invalid_rate': c['invalid'] / c['n'] if c['n'] else 0.0} for t, c in counts.items()},
            'seconds': watch.seconds()}


# ---------------------------------------------------------------------------
# Checkpoints, memory, RNG

def dir_bytes(path):
    return sum(p.stat().st_size for p in Path(path).rglob('*') if p.is_file())


def save_checkpoint(model, ckpt_dir, state):
    """Adapter (``save_pretrained``) + ``trainer_state.pt``, written to a temporary directory and
    renamed into place when complete, so a directory that exists is a complete checkpoint.
    Returns the checkpoint's size in bytes."""
    ckpt_dir = Path(ckpt_dir)
    tmp = ckpt_dir.with_name(ckpt_dir.name + '.tmp')
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir(parents=True)
    model.save_pretrained(str(tmp), safe_serialization=True)
    torch.save(state, tmp / STATE_FILE)
    shutil.rmtree(ckpt_dir, ignore_errors=True)
    tmp.rename(ckpt_dir)
    return dir_bytes(ckpt_dir)


def copy_adapter(ckpt_dir, dest):
    """``best/``: the adapter files of a checkpoint without the trainer state."""
    dest = Path(dest)
    tmp = dest.with_name(dest.name + '.tmp')
    shutil.rmtree(tmp, ignore_errors=True)
    shutil.copytree(ckpt_dir, tmp, ignore=shutil.ignore_patterns(STATE_FILE))
    shutil.rmtree(dest, ignore_errors=True)
    tmp.rename(dest)


def last_checkpoint(out_dir):
    """The highest-numbered complete ``checkpoints/epoch<N>/`` of a run directory, or None."""
    found = []
    for p in (Path(out_dir) / 'checkpoints').glob('epoch*'):
        if p.is_dir() and p.name[5:].isdigit() and (p / STATE_FILE).exists():
            found.append((int(p.name[5:]), p))
    return max(found)[1] if found else None


def trim_train_log(out_dir, step):
    """On resume, drop the ``train_log.jsonl`` lines a killed run wrote after the checkpoint's ``step``
    (the resumed run writes them again) and any line the kill cut short; lines with a step at or
    before the checkpoint are kept as they are.  Returns the number of lines dropped."""
    path = Path(out_dir) / 'train_log.jsonl'
    if not path.exists():
        return 0
    kept, dropped = [], 0
    with open(path, encoding='utf-8') as f:
        for raw in f:
            if not raw.strip():
                continue
            try:
                line = json.loads(raw)
            except ValueError:
                dropped += 1  # a line the kill cut short
                continue
            if line['step'] <= step:
                kept.append(raw.rstrip('\n') + '\n')
            else:
                dropped += 1
    if dropped:
        tmp = path.with_name(path.name + '.tmp')
        with open(tmp, 'w', encoding='utf-8') as f:
            f.writelines(kept)
        os.replace(tmp, path)
    return dropped


def peak_memory_bytes(device):
    if device == 'cuda':
        return int(torch.cuda.max_memory_allocated())
    import resource
    return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * 1024  # Linux reports KiB


def rng_state():
    return {'torch': torch.get_rng_state(), 'python': random.getstate(),
            'cuda': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None}


def set_rng_state(state):
    torch.set_rng_state(state['torch'])
    random.setstate(state['python'])
    if state.get('cuda') is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state['cuda'])


# ---------------------------------------------------------------------------
# The run

def new_record(config, device):
    cuda = device == 'cuda' and torch.cuda.is_available()
    return {'name': config['name'], 'config': config, 'status': 'running', 'error': None,
            'git_commit': common.git_commit(), 'started_at': common.now_iso(), 'finished_at': None, 'seconds': 0.0,
            'device': device, 'gpu_name': torch.cuda.get_device_name(0) if cuda else None,
            'dtype': None, 'model_path': None, 'model_params': None, 'trainable_params': None,
            'examples': {'train': None, 'dropped_too_long': None, 'dev_eval': None},
            'steps': 0, 'epochs_done': 0, 'examples_seen': 0, 'tokens_seen': 0,
            'train_loss': [], 'dev': [], 'selected_epoch': None, 'peak_memory_bytes': None,
            'checkpoint_bytes': None, 'resumed_from': None}


def _replace_epoch(entries, entry):
    """History lists hold one entry per epoch; a resumed partial epoch replaces its earlier entry."""
    entries[:] = [e for e in entries if e['epoch'] != entry['epoch']] + [entry]


def train(args):
    """Run (or resume) the training described by ``args``; returns the final run.json record.

    Every exception is recorded in run.json (``status: failed``, ``error``) and re-raised."""
    config = common.load_config(args.config)
    if args.max_steps is not None:
        config['max_steps'] = args.max_steps
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    device = common.resolve_device(args.device)
    if device == 'cpu':
        torch.set_num_threads(args.threads)
    watch = common.Stopwatch()
    resume_from = last_checkpoint(out) if args.resume else None
    state = None
    if resume_from is not None:
        state = torch.load(resume_from / STATE_FILE, map_location='cpu', weights_only=False)
        dropped_lines = trim_train_log(out, state['step'])
        if dropped_lines:
            print(f'--resume: dropped {dropped_lines} train_log.jsonl line(s) past step {state["step"]}')
    else:  # a fresh start: nothing of an earlier attempt in this directory may survive
        if args.resume:
            print(f'--resume: no complete checkpoint under {out / "checkpoints"}, starting from scratch')
        for name in ('checkpoints', 'best'):
            shutil.rmtree(out / name, ignore_errors=True)
        (out / 'train_log.jsonl').unlink(missing_ok=True)
    record = new_record(config, device)
    seconds_before = 0.0
    if state is not None:
        record['started_at'] = state['started_at']
        record['resumed_from'] = str(resume_from)
        seconds_before = float(state['seconds_before'])

    def elapsed():
        """Cumulative wall-clock seconds over this and every earlier (resumed) invocation."""
        return seconds_before + watch.seconds()

    def write_record():
        record['seconds'] = round(elapsed(), 3)
        record['peak_memory_bytes'] = peak_memory_bytes(device)
        common.write_json(out / 'run.json', record)

    write_record()
    try:
        _train_loop(args, config, out, device, record, state, resume_from, write_record, elapsed)
        record['status'] = 'completed'
    except BaseException as exc:  # noqa: BLE001 - recorded, then re-raised
        record['status'] = 'failed'
        record['error'] = ''.join(traceback.format_exception(exc))[-4000:]
        record['finished_at'] = common.now_iso()
        write_record()
        raise
    record['finished_at'] = common.now_iso()
    write_record()
    return record


def _train_loop(args, config, out, device, record, state, resume_from, write_record, elapsed):
    if device == 'cuda':
        torch.cuda.reset_peak_memory_stats()
    tok = common.load_tokenizer()
    common.check_template(tok)
    legend = common.load_legends(config['data_version'])[config['answer_format']][config['arm']]
    rows = common.training_examples(config)
    items, dropped = tokenise_examples(rows, tok, legend, config)
    if not items:
        raise RuntimeError(f'no training examples: read {len(rows)}, dropped {dropped} longer than max_length')
    dev_rows = select_dev_examples(config, config['dev_eval_examples'])
    record['examples'] = {'train': len(items), 'dropped_too_long': dropped, 'dev_eval': len(dev_rows)}
    model, info = build_model(config, device, adapter_dir=resume_from)
    record.update({k: info[k] for k in ('dtype', 'model_path', 'model_params', 'trainable_params')})
    autocast_dtype = info['autocast_dtype']
    pad_id = tok.pad_token_id

    bs, ga = config['batch_size'], config['grad_accum']
    n_micro = math.ceil(len(items) / bs)
    steps_per_epoch = math.ceil(n_micro / ga)
    planned_steps = steps_per_epoch * config['epochs']
    total_steps = min(planned_steps, config['max_steps']) if config['max_steps'] else planned_steps
    optimizer, scheduler = build_optimizer(model, config, total_steps)

    step = examples_seen = tokens_seen = 0
    history = {'train_loss': [], 'dev': []}
    best = {'epoch': None, 'accuracy': -1.0}
    epoch_loss = {'sum': 0.0, 'tokens': 0}
    start_epoch, start_group = 1, 0
    if state is not None:
        optimizer.load_state_dict(state['optimizer'])
        scheduler.load_state_dict(state['scheduler'])
        set_rng_state(state['rng'])
        step, examples_seen, tokens_seen = state['step'], state['examples_seen'], state['tokens_seen']
        history, best = state['history'], state['best']
        if best['epoch'] is not None:  # the kill may have lost best/; the checkpoint it names always exists
            copy_adapter(out / 'checkpoints' / f"epoch{best['epoch']}", out / 'best')
        if state['epoch_complete']:
            start_epoch = state['epoch'] + 1
        else:
            start_epoch, start_group = state['epoch'], state['batches_done'] // ga
            epoch_loss = state['epoch_loss']
        where = 'complete' if state['epoch_complete'] else f'{state["batches_done"]} micro-batches done'
        print(f'resumed from {resume_from}: step {step}, epoch {state["epoch"]} ({where})')
    record['train_loss'], record['dev'] = history['train_loss'], history['dev']
    record.update(steps=step, epochs_done=state['epoch'] if state else 0, examples_seen=examples_seen,
                  tokens_seen=tokens_seen, selected_epoch=best['epoch'])
    write_record()
    print(f'{config["name"]}: {len(items)} training examples ({dropped} dropped), {len(dev_rows)} dev, '
          f'{steps_per_epoch} steps/epoch, {total_steps} total steps, {device} {info["dtype"]}, '
          f'{info["model_params"]:,} params, {info["trainable_params"]:,} trainable', flush=True)

    def log(epoch, result):
        line = {'step': step, 'epoch': epoch, 'loss': round(result['loss'], 6), 'lr': result['lr'],
                'examples_seen': examples_seen, 'tokens_seen': tokens_seen, 'seconds': round(elapsed(), 3)}
        with open(out / 'train_log.jsonl', 'a', encoding='utf-8') as f:
            f.write(json.dumps(line) + '\n')
        print(f'  step {step} epoch {epoch} loss {result["loss"]:.4f} lr {result["lr"]:.2e} '
              f'grad {result["grad_norm"]:.2f} seen {examples_seen}', flush=True)

    stop = bool(config['max_steps']) and step >= config['max_steps']
    for epoch in range(start_epoch, config['epochs'] + 1):
        if stop:
            break
        perm = epoch_permutation(len(items), config['seed'], epoch)
        micro = [perm[i:i + bs] for i in range(0, len(perm), bs)]
        groups = [micro[i:i + ga] for i in range(0, len(micro), ga)]
        first = start_group if epoch == start_epoch else 0
        if first == 0:
            epoch_loss = {'sum': 0.0, 'tokens': 0}
        batches_done = sum(len(g) for g in groups[:first])
        for g_index in range(first, len(groups)):
            batches = [collate([items[i] for i in idxs], pad_id) for idxs in groups[g_index]]
            result = train_step(model, optimizer, scheduler, batches, device, autocast_dtype)
            step += 1
            examples_seen += result['examples']
            tokens_seen += result['tokens']
            batches_done += len(groups[g_index])
            epoch_loss['sum'] += result['loss'] * result['loss_tokens']
            epoch_loss['tokens'] += result['loss_tokens']
            stop = bool(config['max_steps']) and step >= config['max_steps']
            if step % args.log_every == 0 or g_index == len(groups) - 1 or stop:
                log(epoch, result)
            if stop:
                break
        epoch_complete = batches_done == len(micro)
        mean_loss = epoch_loss['sum'] / max(1, epoch_loss['tokens'])
        _replace_epoch(history['train_loss'], {'epoch': epoch, 'mean_loss': mean_loss})
        dev = dev_evaluate(model, tok, legend, dev_rows, config, device, autocast_dtype, epoch)
        _replace_epoch(history['dev'], dev)
        if dev['accuracy'] >= best['accuracy']:
            best = {'epoch': epoch, 'accuracy': dev['accuracy']}
        ckpt_dir = out / 'checkpoints' / f'epoch{epoch}'
        ckpt_state = {'step': step, 'epoch': epoch, 'epoch_complete': epoch_complete, 'batches_done': batches_done,
                      'examples_seen': examples_seen, 'tokens_seen': tokens_seen, 'epoch_loss': epoch_loss,
                      'history': history, 'best': best, 'total_steps': total_steps, 'config': config,
                      'started_at': record['started_at'], 'seconds_before': elapsed(),
                      'optimizer': optimizer.state_dict(), 'scheduler': scheduler.state_dict(), 'rng': rng_state()}
        ckpt_bytes = save_checkpoint(model, ckpt_dir, ckpt_state)
        if best['epoch'] == epoch:
            copy_adapter(ckpt_dir, out / 'best')
        record.update(steps=step, epochs_done=epoch, examples_seen=examples_seen, tokens_seen=tokens_seen,
                      selected_epoch=best['epoch'], checkpoint_bytes=ckpt_bytes)
        write_record()
        print(f'epoch {epoch}{"" if epoch_complete else " (stopped at max_steps)"}: mean loss {mean_loss:.4f}, '
              f'dev accuracy {dev["accuracy"]:.3f} (invalid {dev["invalid_rate"]:.3f}, {dev["seconds"]:.1f}s), '
              f'checkpoint {ckpt_bytes:,} bytes, best epoch {best["epoch"]}', flush=True)
        if stop:
            break


def parse_args(argv=None):
    p = argparse.ArgumentParser(description='LoRA fine-tuning for one pilot cell (see interfaces.md, Training).')
    p.add_argument('--config', required=True, help='configs/<cell>.json')
    p.add_argument('--out', required=True, help='run directory, runs/<name>')
    p.add_argument('--resume', action='store_true', help='continue from the last checkpoints/epoch<N>/')
    p.add_argument('--device', default='auto', choices=['auto', 'cpu', 'cuda'])
    p.add_argument('--max-steps', type=int, default=None, help='override the config: stop after N optimizer steps')
    p.add_argument('--log-every', type=int, default=10, help='train_log.jsonl line every N steps')
    p.add_argument('--threads', type=int, default=2, help='torch threads on CPU')
    return p.parse_args(argv)


def main(argv=None):
    record = train(parse_args(argv))
    print(json.dumps({k: record[k] for k in ('name', 'status', 'steps', 'epochs_done', 'selected_epoch', 'seconds')}))


if __name__ == '__main__':
    main()

"""Shared pieces of the pretrained-model pilot package.

Everything that two or more scripts must agree on lives here: paths, the arm/task names,
the configuration defaults, the Qwen 3 chat formatting, tokenizer and model loading (with a
tiny randomly-initialised Qwen 3 for CPU smoke tests), token counting and the JSONL helpers.
``interfaces.md`` documents the file formats that the scripts read and write.

Nothing here downloads anything: model and tokenizer directories are local paths (see
``PILOT_MODEL`` / ``PILOT_TOKENIZER``); a download must be allowed explicitly with
``PILOT_ALLOW_DOWNLOAD=1``.
"""
from __future__ import annotations

import copy
import hashlib
import json
import os
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
DATA_DIR = HERE / 'data'
RUNS_DIR = HERE / 'runs'
RESULTS_DIR = HERE / 'results'
CONFIG_DIR = HERE / 'configs'
TINY_MODEL_DIR = HERE / '.tiny_model'  # git-ignored; built on demand by ``tiny_model_dir``

ARMS = ('english', 'json', 'sylang2', 'sylang2g')
TASKS = ('select', 'lookup')
ANSWER_FORMATS = ('json', 'bare')
TEST_PANELS = ('iid', 'wording', 'pair', 'long')
TINY_MODEL = 'tiny-random'

TOKENIZER_ENV = 'PILOT_TOKENIZER'   # directory holding Qwen 3 tokenizer.json + tokenizer_config.json
MODEL_ENV = 'PILOT_MODEL'           # directory holding the base model weights, or 'tiny-random'
PACKET_ENV = 'SYLANG_PACKET'        # the study packet (data.py only, for the english rendering)
DOWNLOAD_ENV = 'PILOT_ALLOW_DOWNLOAD'

# Qwen 3 chat format, non-thinking: identical to
# tokenizer.apply_chat_template(msgs, add_generation_prompt=True, enable_thinking=False).
IM_START, IM_END = '<|im_start|>', '<|im_end|>'
PROMPT_TEMPLATE = (IM_START + 'system\n{legend}' + IM_END + '\n' + IM_START + 'user\n{user}' + IM_END + '\n'
                   + IM_START + 'assistant\n<think>\n\n</think>\n\n')

# The retry request repeats the records and the instruction and appends one of these lines.
FORMAT_REMINDER = {
    ('select', 'json'): 'Your previous answer was not in the required format. Reply with only a JSON list of '
                        '[subject,object] string pairs in record order, for example [["ab1","c2d"]], or [] if none.',
    ('select', 'bare'): 'Your previous answer was not in the required format. Reply with only the subject and '
                        'object of each negated record in record order, separated by single spaces, for example '
                        'ab1 c2d x9 y8; reply with the single word none if there are none.',
    ('lookup', 'json'): 'Your previous answer was not in the required format. Reply with only a JSON object '
                        '{"relation": ..., "polarity": ...} where relation is see, help or contain and polarity '
                        'is positive or negative.',
    ('lookup', 'bare'): 'Your previous answer was not in the required format. Reply with only the relation and '
                        'the polarity separated by one space, for example help negative.',
}

DEFAULT_CONFIG = {
    'name': None,                 # defaults to the config file's stem
    'stage': 'smoke',             # smoke | A | B | C
    'model': TINY_MODEL,          # 'tiny-random' or a local directory (or $PILOT_MODEL when null)
    'arm': 'sylang2',
    'tasks': list(TASKS),
    'answer_format': 'json',
    'data_version': 'v1',
    'train_size': 2000,           # cases; the nested prefix of the seed's training stream
    'seed': 0,                    # data seed (training stream) and training seed
    'lora': {'r': 16, 'alpha': 32, 'dropout': 0.05,
             'target_modules': ['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj']},
    'epochs': 2,
    'lr': 2e-4,
    'batch_size': 16,
    'grad_accum': 1,
    'max_length': 1024,           # prompt + answer tokens; longer examples are dropped and counted
    'warmup_ratio': 0.03,
    'lr_floor': 0.1,              # cosine decay to this fraction of lr
    'weight_decay': 0.0,
    'max_steps': None,            # smoke tests: stop after this many optimizer steps
    'dev_eval_examples': 200,     # dev examples scored by greedy generation at each epoch end
    'max_new_tokens': {'select': 160, 'lookup': 32},
    'timeout_minutes': 90,        # the stage scripts wrap the process in `timeout` with this value
    'dtype': 'auto',              # auto: bf16 on CUDA, float32 on CPU
}


# ---------------------------------------------------------------------------
# Small helpers

def now_iso() -> str:
    return datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')


def read_jsonl(path) -> list[dict]:
    with open(path, encoding='utf-8') as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path, rows) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        for row in rows:
            f.write(json.dumps(row, separators=(',', ':'), ensure_ascii=True) + '\n')


def write_json(path, obj) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(obj, f, indent=1, sort_keys=True)
        f.write('\n')


def sha256_file(path) -> str:
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def git_commit() -> str | None:
    try:
        return subprocess.run(['git', '-C', str(REPO), 'rev-parse', 'HEAD'], capture_output=True, text=True,
                              check=True, timeout=10).stdout.strip()
    except Exception:  # noqa: BLE001 - best effort
        return None


class Stopwatch:
    def __init__(self):
        self.start = time.perf_counter()

    def seconds(self) -> float:
        return time.perf_counter() - self.start


# ---------------------------------------------------------------------------
# Configuration

def _merge(base: dict, override: dict) -> dict:
    out = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def load_config(path) -> dict:
    """A config file is a JSON object; missing keys take ``DEFAULT_CONFIG``."""
    path = Path(path)
    with open(path, encoding='utf-8') as f:
        raw = json.load(f)
    unknown = sorted(set(raw) - set(DEFAULT_CONFIG))
    if unknown:
        raise ValueError(f'{path}: unknown config keys {unknown}')
    config = _merge(DEFAULT_CONFIG, raw)
    config['name'] = config['name'] or path.stem
    if config['arm'] not in ARMS:
        raise ValueError(f'{path}: arm must be one of {ARMS}')
    if config['answer_format'] not in ANSWER_FORMATS:
        raise ValueError(f'{path}: answer_format must be one of {ANSWER_FORMATS}')
    if not config['tasks'] or any(t not in TASKS for t in config['tasks']):
        raise ValueError(f'{path}: tasks must be a non-empty subset of {TASKS}')
    return config


# ---------------------------------------------------------------------------
# Data access

def data_dir(version: str) -> Path:
    return DATA_DIR / version


def load_legends(version: str) -> dict:
    """``{answer_format: {arm: legend text}}`` written by data.py."""
    with open(data_dir(version) / 'legends.json', encoding='utf-8') as f:
        return json.load(f)


def examples_path(version: str, split: str, arm: str, seed: int | None = None) -> Path:
    """train files are per seed; dev and test are shared by every seed."""
    if split == 'train':
        return data_dir(version) / f'train_s{seed}_{arm}.jsonl'
    return data_dir(version) / f'{split}_{arm}.jsonl'


def training_examples(config: dict) -> list[dict]:
    """The nested prefix of ``train_size`` cases of the seed's stream, restricted to the config's tasks."""
    rows = read_jsonl(examples_path(config['data_version'], 'train', config['arm'], config['seed']))
    rows = [r for r in rows if r['case_index'] < config['train_size'] and r['task'] in config['tasks']]
    return rows


def user_text(example: dict, answer_format: str) -> str:
    return example['records'] + example['instruction'][answer_format]


def answer_text(example: dict, answer_format: str) -> str:
    return example['answer'][answer_format]


def prompt_text(legend: str, user: str) -> str:
    return PROMPT_TEMPLATE.format(legend=legend, user=user)


def completion_text(answer: str) -> str:
    """The training target after the prompt: the answer, then the end-of-turn token."""
    return answer + IM_END


def retry_user_text(example: dict, answer_format: str) -> str:
    return user_text(example, answer_format) + '\n' + FORMAT_REMINDER[(example['task'], answer_format)]


# ---------------------------------------------------------------------------
# Tokenizer and model

def tokenizer_dir() -> Path:
    path = os.environ.get(TOKENIZER_ENV)
    if not path:
        raise SystemExit(f'set {TOKENIZER_ENV} to the directory holding the Qwen 3 tokenizer')
    return Path(path)


def load_tokenizer(path=None):
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(str(path or tokenizer_dir()), local_files_only=True)
    if tok.pad_token is None:
        tok.pad_token = '<|endoftext|>'
    return tok


def count_tokens(tok, text: str) -> int:
    return len(tok(text, add_special_tokens=False)['input_ids'])


def decode_output(tok, ids) -> str:
    """The generated ids before the first <|im_end|>, decoded to the text the grader sees; the one
    rule for dev selection (train_lora.py) and evaluation (evaluate.py)."""
    end = tok.convert_tokens_to_ids(IM_END)
    ids = list(ids)
    if end in ids:
        ids = ids[:ids.index(end)]
    return tok.decode(ids, skip_special_tokens=True)


def check_template(tok) -> None:
    """Assert that ``PROMPT_TEMPLATE`` is what the tokenizer's own chat template produces."""
    if getattr(tok, 'chat_template', None) is None:
        return
    msgs = [{'role': 'system', 'content': 'L'}, {'role': 'user', 'content': 'U'}]
    theirs = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    assert theirs == prompt_text('L', 'U'), (theirs, prompt_text('L', 'U'))


def token_breakdown(tok, legend: str, example: dict, answer_format: str) -> dict:
    """Prompt tokens by part. ``template`` is the chat-format overhead (role markers, think block)."""
    system = count_tokens(tok, legend)
    records = count_tokens(tok, example['records'])
    instruction = count_tokens(tok, example['instruction'][answer_format])
    total = count_tokens(tok, prompt_text(legend, user_text(example, answer_format)))
    return {'system': system, 'records': records, 'instruction': instruction,
            'template': total - system - records - instruction, 'prompt_total': total}


def resolve_device(requested: str | None = None) -> str:
    import torch
    if requested and requested != 'auto':
        return requested
    return 'cuda' if torch.cuda.is_available() else 'cpu'


def resolve_dtype(config_dtype: str, device: str):
    import torch
    if config_dtype == 'auto':
        return torch.bfloat16 if device == 'cuda' else torch.float32
    return {'bf16': torch.bfloat16, 'bfloat16': torch.bfloat16, 'fp32': torch.float32,
            'float32': torch.float32, 'fp16': torch.float16, 'float16': torch.float16}[config_dtype]


def tiny_model_dir(seed: int = 0) -> Path:
    """A ~10 M-parameter randomly initialised Qwen 3 (same tokenizer, same architecture) for CPU
    smoke tests; built once, deterministically, under ``.tiny_model``."""
    out = TINY_MODEL_DIR
    if (out / 'config.json').exists() and (out / 'tokenizer.json').exists():
        return out
    import torch
    from transformers import Qwen3Config, Qwen3ForCausalLM
    tok = load_tokenizer()
    torch.manual_seed(seed)
    config = Qwen3Config(vocab_size=151936, hidden_size=64, intermediate_size=128, num_hidden_layers=2,
                         num_attention_heads=4, num_key_value_heads=2, head_dim=16,
                         max_position_embeddings=4096, tie_word_embeddings=True,
                         bos_token_id=None, eos_token_id=tok.convert_tokens_to_ids(IM_END),
                         pad_token_id=tok.pad_token_id)
    model = Qwen3ForCausalLM(config)
    out.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out, safe_serialization=True)
    tok.save_pretrained(out)
    return out


def resolve_model_path(config: dict) -> Path:
    model = config.get('model') or os.environ.get(MODEL_ENV) or TINY_MODEL
    if model == TINY_MODEL:
        return tiny_model_dir()
    path = Path(model)
    if path.is_dir():
        return path
    if os.environ.get(DOWNLOAD_ENV) == '1':
        return path  # a Hub id; from_pretrained may download it
    raise SystemExit(f'model {model!r} is not a local directory; set {DOWNLOAD_ENV}=1 to allow a download')


def load_model(model_path, device: str, dtype, trainable: bool = False):
    import torch
    from transformers import AutoModelForCausalLM
    allow_download = os.environ.get(DOWNLOAD_ENV) == '1'
    model = AutoModelForCausalLM.from_pretrained(str(model_path), torch_dtype=dtype,
                                                 local_files_only=not allow_download)
    model.to(device)
    if not trainable:
        model.eval()
        for p in model.parameters():
            p.requires_grad_(False)
    torch.manual_seed(0)
    return model


def load_adapter(model, adapter_dir):
    """Attach a saved LoRA adapter (a PEFT directory) to a base model, for evaluation."""
    from peft import PeftModel
    model = PeftModel.from_pretrained(model, str(adapter_dir))
    model.eval()
    return model

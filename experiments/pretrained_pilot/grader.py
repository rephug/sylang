"""Deterministic grader for the pilot's two tasks (select, lookup) and two answer formats (json, bare).

``grade(task, answer_format, expected, output)`` returns ``{'correct': bool, 'invalid': bool, 'reason': str}``
(``interfaces.md``). Parsing is strict: the whole output, after stripping surrounding whitespace, must be
exactly one value of the task's shape; anything else is ``invalid`` (and therefore not correct). Equality is
exact on the canonical value, not on the bytes, so ``[["a", "b"]]`` with spaces equals ``[["a","b"]]``.

Reasons: ``ok``; ``wrong`` (valid shape, different value); and the invalid reasons ``empty`` (nothing but
whitespace), ``parse`` (not a value of the format at all: malformed JSON, a bare answer given to the json
format, a JSON answer or punctuation given to the bare format, leading prose with no decodable value),
``extra_text`` (a value of the right format followed or preceded by other text) and ``shape`` (a value of the
format that is not the task's shape: wrong nesting, non-string entries, a missing or extra key, duplicate keys,
an odd number of bare tokens).

Shapes. *select*: json = a list of 2-element lists of strings, ``[]`` for none; bare = ``subject object``
tokens separated by whitespace, each an identifier of the contract's grammar (2-5 characters of ``[a-z0-9]``
with at least one letter), or the single word ``none``. *lookup*: json = an object with exactly the keys
``relation`` and ``polarity`` and string values; bare = exactly two ``[a-z]+`` words. Values outside the
vocabularies (an unknown relation, an identifier that is not in the records) are a valid shape and a ``wrong``
answer. Known limit of the bare format: prose made only of identifier-shaped words (``ab1 c2d is it``) parses
as pairs and is graded ``wrong`` rather than ``invalid``; the json format has no such ambiguity.

``python grader.py items.jsonl`` grades a JSONL file whose lines carry ``task``, ``answer_format``,
``expected`` and ``output`` and prints the counts (optional helper).
"""
from __future__ import annotations

import json
import re
import sys

TASKS = ('select', 'lookup')
ANSWER_FORMATS = ('json', 'bare')
LOOKUP_KEYS = ('relation', 'polarity')
NONE_WORD = 'none'
_IDENTIFIER = re.compile(r'(?=[0-9]*[a-z])[a-z0-9]{2,5}')  # contract.md: 2-5 of [a-z0-9], at least one letter
_WORD = re.compile(r'[a-z]+')
_OPENER = {'select': '[', 'lookup': '{'}


class _DuplicateKey(ValueError):
    pass


def _pairs_without_duplicates(pairs):
    keys = [k for k, _ in pairs]
    if len(keys) != len(set(keys)):
        raise _DuplicateKey('duplicate key')
    return dict(pairs)


_STRICT = json.JSONDecoder(object_pairs_hook=_pairs_without_duplicates)
_LOOSE = json.JSONDecoder()


# ---------------------------------------------------------------------------
# Parsing to the canonical value: returns (value, 'ok') or (None, reason)

def _shape(task, value):
    if task == 'select':
        if not isinstance(value, list):
            return None, 'shape'
        pairs = []
        for item in value:
            if not (isinstance(item, list) and len(item) == 2 and all(isinstance(x, str) for x in item)):
                return None, 'shape'
            pairs.append((item[0], item[1]))
        return tuple(pairs), 'ok'
    if not isinstance(value, dict) or tuple(sorted(value)) != tuple(sorted(LOOKUP_KEYS)):
        return None, 'shape'
    if not all(isinstance(value[k], str) for k in LOOKUP_KEYS):
        return None, 'shape'
    return (value['relation'], value['polarity']), 'ok'


def _parse_json(task, text):
    try:
        value, end = _STRICT.raw_decode(text)
    except _DuplicateKey:
        return None, 'shape'
    except ValueError:
        # No JSON value at the start. If a value of the task's shape starts later, the output is prose
        # around a value (extra_text); otherwise it is not JSON at all (parse).
        opener = _OPENER[task]
        at = text.find(opener)
        while at != -1:
            try:
                value, _ = _LOOSE.raw_decode(text, at)
            except ValueError:
                value = None
            if value is not None and _shape(task, value)[0] is not None:
                return None, 'extra_text'
            at = text.find(opener, at + 1)
        return None, 'parse'
    if text[end:].strip():
        return None, 'extra_text'
    return _shape(task, value)


def _parse_bare(task, text):
    tokens = text.split()
    if task == 'select':
        if any(not _IDENTIFIER.fullmatch(t) for t in tokens):
            return None, 'parse'
        if tokens == [NONE_WORD]:
            return (), 'ok'
        if NONE_WORD in tokens or len(tokens) % 2:
            return None, 'shape'
        return tuple(zip(tokens[0::2], tokens[1::2])), 'ok'
    if any(not _WORD.fullmatch(t) for t in tokens):
        return None, 'parse'
    if len(tokens) != 2:
        return None, 'shape'
    return (tokens[0], tokens[1]), 'ok'


_FENCE = re.compile(r'\A```[A-Za-z0-9_-]*[ \t]*\r?\n(.*?)\r?\n?```\Z', re.S)


def unfence(text: str) -> tuple[str, bool]:
    """Strip one Markdown code fence wrapping the whole output (```json ... ```), a chat-model habit
    tolerated by the pre-registration amendment of 2026-10-22; returns (inner text, fenced)."""
    m = _FENCE.match(text.strip())
    return (m.group(1).strip(), True) if m else (text, False)


def parse(task, answer_format, text):
    """Canonical value of an answer text, or ``(None, reason)`` when it is invalid. A single code
    fence around the whole output is removed first (see ``unfence``)."""
    if task not in TASKS:
        raise ValueError(f'task must be one of {TASKS}, not {task!r}')
    if answer_format not in ANSWER_FORMATS:
        raise ValueError(f'answer_format must be one of {ANSWER_FORMATS}, not {answer_format!r}')
    text = text.strip() if isinstance(text, str) else ''
    text, _ = unfence(text)
    if not text:
        return None, 'empty'
    return (_parse_json if answer_format == 'json' else _parse_bare)(task, text)


def grade(task: str, answer_format: str, expected: str, output: str) -> dict:
    """``{'correct', 'invalid', 'reason'}`` for one model output against the canonical expected answer."""
    want, reason = parse(task, answer_format, expected)
    if want is None:
        raise ValueError(f'expected answer is not a canonical {task}/{answer_format} answer ({reason}): {expected!r}')
    fenced = unfence(output.strip() if isinstance(output, str) else '')[1]
    got, reason = parse(task, answer_format, output)
    if got is None:
        return {'correct': False, 'invalid': True, 'reason': reason, 'fenced': fenced}
    if got == want:
        return {'correct': True, 'invalid': False, 'reason': 'ok', 'fenced': fenced}
    return {'correct': False, 'invalid': False, 'reason': 'wrong', 'fenced': fenced}


# ---------------------------------------------------------------------------
# Optional CLI: grade a JSONL of {task, answer_format, expected, output}

def main(argv=None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if not argv or argv[0] in ('-h', '--help'):
        print('usage: grader.py <jsonl with task, answer_format, expected, output per line>', file=sys.stderr)
        return 0 if argv else 2
    if len(argv) != 1:
        print('usage: grader.py items.jsonl   (lines with task, answer_format, expected, output)', file=sys.stderr)
        return 2
    n = correct = invalid = 0
    reasons: dict[str, int] = {}
    with open(argv[0], encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            result = grade(row['task'], row['answer_format'], row['expected'], row.get('output') or '')
            n += 1
            correct += result['correct']
            invalid += result['invalid']
            reasons[result['reason']] = reasons.get(result['reason'], 0) + 1
    print(json.dumps({'n': n, 'correct': correct, 'invalid': invalid, 'reasons': reasons}))
    return 0


if __name__ == '__main__':
    sys.exit(main())

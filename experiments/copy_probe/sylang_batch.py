"""SYLANG-BATCH codec: the v1 wire format with its header, and the opt-in header-free v2.

    python -I sylang_batch.py encode --version 2 [--style codes|words] [--marker] < records.jsonl > out.txt
    python -I sylang_batch.py decode [--info] < text > records.jsonl
    python -I sylang_batch.py convert --to 2|1 [--marker] < text > text

This is an independent implementation written from the wire format alone. Every
rule below was derived empirically from the format's observable output (all 216
feature combinations in both styles, and a scan of every Unicode code point in
the first and a later position of an identifier); nothing here is copied from
another codec. ``encode(records, version=1)`` is byte-identical to the project's
reference v1 rendering (style ``codes`` or ``words``, policy ``selective``, no
dictionary) for every input both accept, and ``decode`` is the exact inverse of
``encode`` for both versions and both styles.

Records
=======

A record is a predicate statement with seven fields:

===========  ==========================================================  =============
field        values                                                      default
===========  ==========================================================  =============
subject      an identifier (see below)                                   required
object       an identifier                                               required
relation     ``see`` | ``help`` | ``contain``                            required
polarity     ``positive`` | ``negative``                                 ``positive``
tense        ``past`` | ``present`` | ``future``                         ``present``
aspect       ``simple`` | ``progressive`` | ``completed``                ``simple``
evidence     ``unspecified`` | ``direct`` | ``reported`` | ``inferred``  ``unspecified``
===========  ==========================================================  =============

``encode`` accepts dicts with these keys (missing non-required keys take their
defaults) plus an optional ``kind`` that must be ``pred``. ``decode`` returns dicts
with all seven keys, in the order above. A batch holds one or more records.

Record line (both versions)
===========================

One line per record, ``<subject><TAB><middle><TAB><object>``, terminated by LF.

``codes`` middle: one relation letter, UPPERCASE when the polarity is negative,
then optional single letters for tense, aspect and evidence, in that order, each
present only when the feature is not its default:

    relation  see ``s``  help ``h``  contain ``c``   (``S`` ``H`` ``C`` when negative)
    tense     past ``p``  future ``f``                 (present: nothing)
    aspect    progressive ``g``  completed ``c``       (simple: nothing)
    evidence  direct ``d``  reported ``r``  inferred ``i``   (unspecified: nothing)

So ``s`` is "see, positive, present, simple, unspecified", ``Hfgi`` is "help,
negative, future, progressive, inferred" and ``cc`` is "contain, present, completed".

``words`` middle: the relation word, then ``negative`` when the polarity is
negative, then the tense word when it is not ``present``, the aspect word when it
is not ``simple`` and the evidence word when it is not ``unspecified``, in that
order, separated by single spaces: ``see``, ``help negative future progressive
inferred``, ``contain completed``.

Identifiers
===========

Any non-empty string without a tab or a line feed and without a leading ``@`` is
accepted (the three are validation errors: ``@`` opens a dictionary reference in
the wire format, and tab and LF are the field and record separators). The
``selective`` policy spells an identifier in one of two modes; the decoder
reverses the escapes and requires the spelling to be canonical (an identifier
spelled with an escape this encoder would not emit is a ``ValueError``).

*Light mode* applies to an identifier whose light spelling (below) is in Unicode
NFC (``unicodedata.normalize('NFC', spelling) == spelling``) and that contains no
``<``, no ``[`` and no unassigned code point (general category ``Cn``, which
includes the noncharacters). The NFC test is on the spelling, not on the raw
identifier: a combining mark that follows an escaped control attaches to the
escape's last letter (``\\r`` followed by U+0323 reads as ``r`` + U+0323, which
composes to U+1E5B), so U+000D U+0323 is spelled in full mode while U+0323 alone,
or after a letter it does not compose with, stays in light mode. Light mode
escapes only:

- the C0 controls: U+0008 ``\\b``, U+000C ``\\f``, U+000D ``\\r``, the others as
  ``\\u00XX`` (lower-case hex); tab and LF would be ``\\t`` and ``\\n`` in the wire
  format, but this codec rejects them instead;
- ``"`` as ``\\"`` and ``\\`` as ``\\\\``;
- a leading ``@`` as ``\\u0040`` (rejected by this codec).

Everything else is written as it is: DEL, the C1 controls, spaces, ``]``, ``>``,
``#``, a non-leading ``@``, accented letters, CJK, emoji, private-use and format
characters.

*Full mode* applies to every other identifier: one whose light spelling is not
in NFC (a decomposed accent such as ``e`` + U+0301, a Hangul jamo sequence, a
singleton such as U+2126 OHM SIGN or U+2000, a mark after an escape as above), or
that contains ``<``, ``[`` or an unassigned code point. It applies the light-mode escapes and additionally writes ``<`` as
``\\u003c``, ``[`` as ``\\u005b`` and every code point from U+007F upwards as
``\\uXXXX``, or as a ``\\uXXXX\\uXXXX`` surrogate pair above the BMP, so the wire
spelling is printable ASCII and NFC-stable.

In both modes the identifier ``if`` (exactly) is written ``\\u0069f``, because
``if`` opens an ``if`` record in the wire format, and unpaired surrogates are a
validation error.

Version 1 (``SYLANG-BATCH`` header)
===================================

Line 1 is ``SYLANG-BATCH<TAB>1<TAB><style><TAB>selective<TAB><N><TAB><R>`` where
style is ``codes`` or ``words``, N is the number of dictionary entries (always 0
here) and R the number of records. The record lines follow. The text ends with
exactly one LF. This is the existing format: every existing input decodes.

Version 2 (header-free, opt-in)
===============================

The record lines alone, byte-identical to version 1's, with no header line; the
text ends with exactly one LF. The style is implied by the record lines. With
``marker=True`` (``--marker``) the text starts with the line ``#sylang2``. The
marker is optional and costs tokens: measured with the Qwen 3 tokenizer.json
used by ``tokens.py``, ``#sylang2\\n`` is 6 tokens (``#``, ``s``, ``yl``, ``ang``, ``2``,
``\\n``), against 15 for a v1 header line. Default ``marker=False``.

Version 2 is opt-in: ``encode`` defaults to ``version=1``. Decoders accept both:
``decode`` recognises v1 by a first line starting with ``SYLANG-BATCH<TAB>``, v2
with the marker by a first line equal to ``#sylang2``, and otherwise v2 without
the marker. For that reason the subject of the first record of a v2 text without
the marker must not be the string ``SYLANG-BATCH`` (``encode`` rejects it; with
the marker, or under a v1 header, it is allowed).

Validation in ``decode``
========================

``ValueError`` for empty text, text not ending in LF, a record line without
exactly three tab-separated fields, an unknown middle (unknown letter or word,
wrong order, repeated feature, a default spelled out, double spaces), an empty
identifier, a non-canonical or unknown escape, a raw character that the policy
would have escaped (including CR, so CRLF line endings are rejected), a v1 header
with the wrong field count, a version other than 1, an unknown style or a style
that does not match the record lines, an R that is not the record count or is 0,
mixed styles in one text, a v2 text with no records, or an identifier with a
leading ``@`` (``\\u0040``) or with tab or LF escapes. ``NotImplementedError`` for
a dictionary (N > 0, interning), a policy other than ``selective``, an ``if``
record (a line whose first field is the bare word ``if``) and a record ``kind``
other than ``pred``.

Not supported: interning (dictionaries, ``@`` references), ``if`` records,
policies other than ``selective``, and identifiers containing tab or LF or
starting with ``@``. Unlike the reference renderer, this codec puts no cap on the
number of records in a batch (the reference accepts 1..1000) or on identifier
length (the reference accepts 1..256 Unicode scalars).

API
===

    encode(records, *, style='codes', version=1, marker=False) -> str
    decode(text) -> (records, info)    # info: {'version', 'style', 'records', 'marker'}
    encode_middle(record, style) -> str
    parse_middle(middle) -> (style, features)
    escape_identifier(identifier) -> str
    unescape_identifier(field) -> str
"""

from __future__ import annotations

import argparse
import json
import sys
import unicodedata

__all__ = ['encode', 'decode', 'encode_middle', 'parse_middle', 'escape_identifier', 'unescape_identifier',
           'FIELDS', 'DEFAULTS', 'RELATIONS', 'POLARITIES', 'TENSES', 'ASPECTS', 'EVIDENCES', 'STYLES',
           'HEADER_TAG', 'MARKER', 'POLICY']

HEADER_TAG = 'SYLANG-BATCH'
MARKER = '#sylang2'
POLICY = 'selective'
STYLES = ('codes', 'words')
VERSIONS = (1, 2)

RELATIONS = ('see', 'help', 'contain')
POLARITIES = ('positive', 'negative')
TENSES = ('past', 'present', 'future')
ASPECTS = ('simple', 'progressive', 'completed')
EVIDENCES = ('unspecified', 'direct', 'reported', 'inferred')
FIELDS = ('subject', 'object', 'relation', 'polarity', 'tense', 'aspect', 'evidence')
DEFAULTS = {'polarity': 'positive', 'tense': 'present', 'aspect': 'simple', 'evidence': 'unspecified'}
VALUES = {'relation': RELATIONS, 'polarity': POLARITIES, 'tense': TENSES, 'aspect': ASPECTS, 'evidence': EVIDENCES}

# codes style: letter tables (the default of each feature has no letter)
RELATION_LETTER = {'see': 's', 'help': 'h', 'contain': 'c'}
TENSE_LETTER = {'present': '', 'past': 'p', 'future': 'f'}
ASPECT_LETTER = {'simple': '', 'progressive': 'g', 'completed': 'c'}
EVIDENCE_LETTER = {'unspecified': '', 'direct': 'd', 'reported': 'r', 'inferred': 'i'}
_RELATION_OF = {v: k for k, v in RELATION_LETTER.items()}
_TENSE_OF = {v: k for k, v in TENSE_LETTER.items() if v}
_ASPECT_OF = {v: k for k, v in ASPECT_LETTER.items() if v}
_EVIDENCE_OF = {v: k for k, v in EVIDENCE_LETTER.items() if v}

# identifier escaping
_SHORT_ESCAPE = {'\x08': '\\b', '\x0c': '\\f', '\n': '\\n', '\r': '\\r', '\t': '\\t', '"': '\\"', '\\': '\\\\'}
_SHORT_UNESCAPE = {'b': '\x08', 'f': '\x0c', 'n': '\n', 'r': '\r', 't': '\t', '"': '"', '\\': '\\'}
_ALWAYS_ESCAPED = set('"\\')  # escaped in both modes (with the C0 controls and a leading '@')
_MARKUP = set('<[')  # their presence selects full mode, where they are escaped too
_HEX = frozenset('0123456789abcdefABCDEF')
_FULL_FROM = 0x7f  # full mode: every code point from here up is escaped


def _light(identifier: str) -> str:
    """The light-mode spelling: C0 controls, '"' and '\\' escaped, everything else as it is."""
    return ''.join(_SHORT_ESCAPE.get(ch) or _unit(ord(ch)) if ord(ch) < 0x20 or ch in _ALWAYS_ESCAPED else ch
                   for ch in identifier)


def _full_mode(identifier: str, light: str) -> bool:
    """True when the selective policy escapes the identifier fully (all non-printable-ASCII).

    The NFC test is on the light spelling, not the raw identifier: a combining mark right after
    an escaped control reads as attached to the escape's last letter (``\\r`` followed by U+0323
    reads as ``r`` + U+0323, which composes to U+1E5B), which makes that spelling unstable."""
    return (unicodedata.normalize('NFC', light) != light
            or any(ch in _MARKUP or unicodedata.category(ch) == 'Cn' for ch in identifier))


class _Problem(ValueError):
    """A validation error with a message that names no identifier text."""


# ---------------------------------------------------------------- identifiers

def _unit(code: int) -> str:
    """``\\uXXXX`` for a BMP code point, a surrogate pair above the BMP."""
    if code < 0x10000:
        return '\\u%04x' % code
    high, low = divmod(code - 0x10000, 0x400)
    return '\\u%04x\\u%04x' % (0xd800 + high, 0xdc00 + low)


def _validate_identifier(identifier, what: str = 'identifier') -> None:
    if not isinstance(identifier, str):
        raise TypeError(f'{what} must be a str, not {type(identifier).__name__}')
    if identifier == '':
        raise ValueError(f'{what} must not be empty')
    if '\t' in identifier or '\n' in identifier:
        raise ValueError(f'{what} must not contain a tab or a line feed')
    if identifier[0] == '@':
        raise ValueError(f"{what} must not start with '@' (dictionary references are not supported)")
    try:
        identifier.encode('utf-8')
    except UnicodeEncodeError:
        raise ValueError(f'{what} must not contain unpaired surrogates') from None


def escape_identifier(identifier: str) -> str:
    """The wire spelling of an identifier under the selective policy (see the module doc)."""
    _validate_identifier(identifier)
    if identifier == 'if':
        return '\\u0069f'
    light = _light(identifier)
    if not _full_mode(identifier, light):
        return light
    out = []
    for ch in identifier:
        code = ord(ch)
        if code < 0x20 or ch in _ALWAYS_ESCAPED:
            out.append(_SHORT_ESCAPE.get(ch) or _unit(code))
        elif code >= _FULL_FROM or ch in _MARKUP:
            out.append(_unit(code))
        else:
            out.append(ch)
    return ''.join(out)


def _unescape(field: str) -> str:
    out = []
    i, n = 0, len(field)
    while i < n:
        ch = field[i]
        if ch != '\\':
            out.append(ch)
            i += 1
            continue
        if i + 1 >= n:
            raise ValueError('dangling backslash in identifier')
        kind = field[i + 1]
        if kind in _SHORT_UNESCAPE:
            out.append(_SHORT_UNESCAPE[kind])
            i += 2
            continue
        if kind != 'u':
            raise ValueError('unknown escape in identifier')
        code = _hex4(field, i + 2)
        i += 6
        if 0xdc00 <= code <= 0xdfff:
            raise ValueError('unpaired low surrogate escape in identifier')
        if 0xd800 <= code <= 0xdbff:
            if field[i:i + 2] != '\\u':
                raise ValueError('unpaired high surrogate escape in identifier')
            low = _hex4(field, i + 2)
            if not 0xdc00 <= low <= 0xdfff:
                raise ValueError('unpaired high surrogate escape in identifier')
            i += 6
            code = 0x10000 + ((code - 0xd800) << 10) + (low - 0xdc00)
        out.append(chr(code))
    return ''.join(out)


def _hex4(field: str, at: int) -> int:
    digits = field[at:at + 4]
    if len(digits) != 4 or not all(d in _HEX for d in digits):
        raise ValueError('malformed \\u escape in identifier')
    return int(digits, 16)


def unescape_identifier(field: str) -> str:
    """The identifier spelled by a wire field; raises ValueError unless the spelling is canonical."""
    if not isinstance(field, str):
        raise TypeError(f'field must be a str, not {type(field).__name__}')
    if field == '':
        raise ValueError('empty identifier')
    if field == 'if':
        raise NotImplementedError("'if' records are not supported")
    if field[0] == '@':
        raise ValueError("a field starting with '@' is a dictionary reference; interning is not supported")
    raw = _unescape(field)
    if raw == '':
        raise ValueError('empty identifier')
    if '\t' in raw or '\n' in raw:
        raise ValueError('identifiers containing a tab or a line feed are not supported')
    if raw[0] == '@':
        raise ValueError("identifiers starting with '@' are not supported")
    if escape_identifier(raw) != field:
        hint = ' (CRLF line endings are not accepted)' if raw.endswith('\r') else ''
        raise ValueError('identifier is not spelled canonically for the selective policy' + hint)
    return raw


# -------------------------------------------------------------------- records

def _normalize_record(record, index: int) -> dict:
    """A validated record with all seven keys, in FIELDS order."""
    if not isinstance(record, dict):
        raise TypeError(f'record {index}: expected a dict, not {type(record).__name__}')
    unknown = set(record) - set(FIELDS) - {'kind'}
    if unknown:
        raise ValueError(f'record {index}: unknown key(s) {sorted(unknown)}')
    if record.get('kind', 'pred') != 'pred':
        raise NotImplementedError(f'record {index}: only kind "pred" is supported')
    out = {}
    for field in ('subject', 'object'):
        if field not in record:
            raise ValueError(f'record {index}: missing {field}')
        try:
            _validate_identifier(record[field], field)
        except (TypeError, ValueError) as error:
            raise type(error)(f'record {index}: {error}') from None
        out[field] = record[field]
    if 'relation' not in record:
        raise ValueError(f'record {index}: missing relation')
    for field, allowed in VALUES.items():
        value = record[field] if field in record else DEFAULTS[field]
        if value not in allowed:
            raise ValueError(f'record {index}: {field} must be one of {allowed}')
        out[field] = value
    return out


def encode_middle(record: dict, style: str = 'codes') -> str:
    """The middle field of a (validated or default-free) record in the given style."""
    rec = _normalize_record(record, 0)
    if style == 'codes':
        letter = RELATION_LETTER[rec['relation']]
        if rec['polarity'] == 'negative':
            letter = letter.upper()
        return letter + TENSE_LETTER[rec['tense']] + ASPECT_LETTER[rec['aspect']] + EVIDENCE_LETTER[rec['evidence']]
    if style == 'words':
        parts = [rec['relation']]
        if rec['polarity'] == 'negative':
            parts.append('negative')
        for field in ('tense', 'aspect', 'evidence'):
            if rec[field] != DEFAULTS[field]:
                parts.append(rec[field])
        return ' '.join(parts)
    raise ValueError(f'unknown style {style!r}; expected one of {STYLES}')


def _parse_codes(middle: str) -> dict | None:
    """Features of a codes middle (``[shcSHC][pf]?[gc]?[dri]?``), or None when it is not one."""
    if middle == '' or middle[0].lower() not in _RELATION_OF:
        return None
    features = {'relation': _RELATION_OF[middle[0].lower()],
                'polarity': 'negative' if middle[0].isupper() else 'positive'}
    rest = middle[1:]
    for field, table in (('tense', _TENSE_OF), ('aspect', _ASPECT_OF), ('evidence', _EVIDENCE_OF)):
        if rest and rest[0] in table:
            features[field] = table[rest[0]]
            rest = rest[1:]
        else:
            features[field] = DEFAULTS[field]
    return None if rest else features


def _parse_words(middle: str) -> dict | None:
    """Features of a words middle (relation [negative] [tense] [aspect] [evidence]), or None."""
    parts = middle.split(' ')
    if parts[0] not in RELATIONS or '' in parts:
        return None
    features = {'relation': parts[0]}
    rest = parts[1:]
    if rest and rest[0] == 'negative':
        features['polarity'] = 'negative'
        rest = rest[1:]
    else:
        features['polarity'] = 'positive'
    for field in ('tense', 'aspect', 'evidence'):
        if rest and rest[0] in VALUES[field] and rest[0] != DEFAULTS[field]:
            features[field] = rest[0]
            rest = rest[1:]
        else:
            features[field] = DEFAULTS[field]
    return None if rest else features


def parse_middle(middle: str) -> tuple[str, dict]:
    """(style, features) for a middle field; ValueError when it is not a valid middle in either style.

    The grammars are disjoint: a codes middle is 1..4 letters from the tables, a words middle
    starts with a whole relation word, so a middle parses in at most one style."""
    features = _parse_codes(middle)
    if features is not None:
        return 'codes', features
    features = _parse_words(middle)
    if features is not None:
        return 'words', features
    raise ValueError('unknown middle field (not a valid codes or words middle)')


def _record_line(rec: dict, style: str) -> str:
    return escape_identifier(rec['subject']) + '\t' + encode_middle(rec, style) + '\t' + escape_identifier(rec['object'])


def _parse_line(line: str, number: int) -> tuple[str, dict]:
    parts = line.split('\t')
    if len(parts) != 3:
        raise ValueError(f'record line {number}: expected 3 tab-separated fields, found {len(parts)}')
    try:
        subject = unescape_identifier(parts[0])
        style, features = parse_middle(parts[1])
        obj = unescape_identifier(parts[2])
    except (ValueError, NotImplementedError) as error:
        raise type(error)(f'record line {number}: {error}') from None
    record = {'subject': subject, 'object': obj}
    record.update((field, features[field]) for field in FIELDS[2:])
    return style, record


# --------------------------------------------------------------- encode/decode

def encode(records, *, style: str = 'codes', version: int = 1, marker: bool = False) -> str:
    """Render a batch of records. ``version=1`` (default) writes the SYLANG-BATCH header line;
    ``version=2`` writes the header-free text, with the ``#sylang2`` marker line when ``marker``."""
    if style not in STYLES:
        raise ValueError(f'unknown style {style!r}; expected one of {STYLES}')
    if version not in VERSIONS:
        raise ValueError(f'unknown version {version!r}; expected one of {VERSIONS}')
    if marker and version != 2:
        raise ValueError('the #sylang2 marker is only defined for version 2')
    records = list(records)
    if not records:
        raise ValueError('a batch must hold at least one record')
    normalized = [_normalize_record(record, index) for index, record in enumerate(records)]
    lines = [_record_line(rec, style) for rec in normalized]
    if version == 1:
        header = '\t'.join((HEADER_TAG, '1', style, POLICY, '0', str(len(lines))))
        return header + '\n' + ''.join(line + '\n' for line in lines)
    if not marker and normalized[0]['subject'] == HEADER_TAG:
        raise ValueError(f'version 2 without the marker: the first subject must not be {HEADER_TAG!r} '
                         '(its line would read as a v1 header)')
    return (MARKER + '\n' if marker else '') + ''.join(line + '\n' for line in lines)


def decode(text: str) -> tuple[list[dict], dict]:
    """Parse a v1 or v2 text. Returns (records, info) with info = {'version', 'style', 'records', 'marker'}."""
    if not isinstance(text, str):
        raise TypeError(f'text must be a str, not {type(text).__name__}')
    if text == '':
        raise ValueError('empty text')
    if not text.endswith('\n'):
        raise ValueError('text must end with a line feed')
    lines = text[:-1].split('\n')
    header_style = None
    marker = False
    if lines[0].startswith(HEADER_TAG + '\t'):
        version = 1
        fields = lines[0].split('\t')
        if len(fields) != 6:
            raise ValueError(f'malformed {HEADER_TAG} header: expected 6 fields, found {len(fields)}')
        _, ver, header_style, policy, entries, count = fields
        if ver != '1':
            raise ValueError(f'unsupported {HEADER_TAG} version {ver!r}')
        if header_style not in STYLES:
            raise ValueError(f'unknown style {header_style!r} in header; expected one of {STYLES}')
        if policy != POLICY:
            raise NotImplementedError(f'policy {policy!r} is not supported; only {POLICY!r}')
        if not entries.isdecimal() or not count.isdecimal():
            raise ValueError('malformed header: N and R must be decimal integers')
        if int(entries) > 0:
            raise NotImplementedError('dictionary entries (interning) are not supported')
        body = lines[1:]
        if int(count) != len(body):
            raise ValueError(f'header says {int(count)} records, found {len(body)} record lines')
    elif lines[0] == MARKER:
        version, marker = 2, True
        body = lines[1:]
    else:
        version = 2
        body = lines
    if not body:
        raise ValueError('a batch must hold at least one record')
    style = header_style
    records = []
    for number, line in enumerate(body, 1):
        line_style, record = _parse_line(line, number)
        if style is None:
            style = line_style
        elif line_style != style:
            where = 'the header' if header_style else 'the first record'
            raise ValueError(f'record line {number} is in style {line_style!r} but {where} is {style!r}')
        records.append(record)
    return records, {'version': version, 'style': style, 'records': len(records), 'marker': marker}


# ------------------------------------------------------------------------ CLI

def _read_records(data: str) -> list:
    stripped = data.strip()
    if stripped.startswith('['):
        return json.loads(stripped)
    return [json.loads(line) for line in data.split('\n') if line.strip()]


def _write_records(records: list) -> str:
    return ''.join(json.dumps(record, ensure_ascii=False) + '\n' for record in records)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description='SYLANG-BATCH v1/v2 codec (stdin to stdout)')
    sub = parser.add_subparsers(dest='command', required=True)
    enc = sub.add_parser('encode', help='JSON records (JSONL or a JSON array) on stdin -> text')
    enc.add_argument('--version', type=int, choices=VERSIONS, default=1, help='1: header (default); 2: header-free')
    enc.add_argument('--style', choices=STYLES, default='codes')
    enc.add_argument('--marker', action='store_true', help='version 2 only: start with the #sylang2 line')
    dec = sub.add_parser('decode', help='v1 or v2 text on stdin -> JSONL records')
    dec.add_argument('--info', action='store_true', help='also print the info dict to stderr')
    conv = sub.add_parser('convert', help='re-encode a v1 or v2 text in the other version, same style')
    conv.add_argument('--to', type=int, choices=VERSIONS, required=True)
    conv.add_argument('--marker', action='store_true', help='with --to 2: start with the #sylang2 line')
    args = parser.parse_args(argv)
    try:
        data = sys.stdin.buffer.read().decode('utf-8')
        if args.command == 'encode':
            out = encode(_read_records(data), style=args.style, version=args.version, marker=args.marker)
        elif args.command == 'decode':
            records, info = decode(data)
            out = _write_records(records)
            if args.info:
                print(json.dumps(info), file=sys.stderr)
        else:
            records, info = decode(data)
            out = encode(records, style=info['style'], version=args.to, marker=args.marker and args.to == 2)
    except (ValueError, NotImplementedError, TypeError, UnicodeDecodeError) as error:
        print(f'error: {error}', file=sys.stderr)
        return 2
    sys.stdout.buffer.write(out.encode('utf-8'))
    sys.stdout.buffer.flush()
    return 0


if __name__ == '__main__':
    sys.exit(main())

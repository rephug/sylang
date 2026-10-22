"""Tests for sylang_batch: letter tables, round trips, malformed input, and parity with the
reference v1 renderer when SYLANG_PACKET names a packet directory.

    python -I test_sylang_batch.py -v
    SYLANG_PACKET=<packet> python -I test_sylang_batch.py -v

No test prints an identifier: assertions report only positions, lengths and counts.
"""

from __future__ import annotations

import itertools
import os
import random
import string
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import sylang_batch as sb  # noqa: E402

COMBINATIONS = list(itertools.product(sb.RELATIONS, sb.POLARITIES, sb.TENSES, sb.ASPECTS, sb.EVIDENCES))
ALNUM = string.ascii_letters + string.digits
# Characters that exercise every escaping rule: controls, the always-escaped punctuation, a
# non-leading '@', combining marks (NFC-unstable after a base letter), NFC singletons, Hangul
# jamo, an unassigned code point, a noncharacter, DEL, a C1 control, format characters,
# private use, a non-BMP character, and plain non-ASCII letters.
ODD = ['\x00', '\x08', '\x0b', '\x0c', '\r', '\x1f', '"', '\\', '<', '[', ']', '@', '#', ' ', '~',
       '\u0300', '\u0301', '\u030b', '\u0323', '\u0338', '\u2126', '\u212b', '\u2000', '\u0958', '\u1100',
       '\u1161', '\u11a8', '\uac00', '\u304b', '\u3099', '\u0378', '\ufffe', '\U0001fffe', '\x7f', '\x80',
       '\u200d', '\ufeff', '\ue000', '\U0001f600', '\u00e9', '\u65e5', 'e', 'a', 'o', '=', 'i', 'f']


def record_of(subject, obj, combination):
    relation, polarity, tense, aspect, evidence = combination
    return {'subject': subject, 'object': obj, 'relation': relation, 'polarity': polarity,
            'tense': tense, 'aspect': aspect, 'evidence': evidence}


def random_identifier(rng, alphabet=ALNUM, longest=8):
    return ''.join(rng.choice(alphabet) for _ in range(rng.randint(1, longest)))


def random_batch(rng, size, alphabet=ALNUM):
    out = []
    for _ in range(size):
        rec = record_of(random_identifier(rng, alphabet), random_identifier(rng, alphabet), rng.choice(COMBINATIONS))
        if alphabet is not ALNUM:  # keep the identifiers inside the accepted domain
            for key in ('subject', 'object'):
                while rec[key][0] == '@' or rec[key] in ('\t', '\n') or rec[key] == sb.HEADER_TAG:
                    rec[key] = random_identifier(rng, alphabet)
        out.append(rec)
    return out


def with_defaults_dropped(record, rng):
    """The same record with some default-valued features left out (encode must fill them in)."""
    out = dict(record)
    for key, default in sb.DEFAULTS.items():
        if out[key] == default and rng.random() < 0.5:
            del out[key]
    return out


class NoLeak(unittest.TestCase):
    """Assertions whose failure messages carry no identifier text."""

    def same_text(self, actual, expected, where):
        if actual != expected:
            offset = next((i for i, (a, b) in enumerate(zip(actual, expected)) if a != b),
                          min(len(actual), len(expected)))
            self.fail(f'{where}: texts differ at offset {offset} (lengths {len(actual)} vs {len(expected)})')

    def same_records(self, actual, expected, where):
        if len(actual) != len(expected):
            self.fail(f'{where}: {len(actual)} records decoded, expected {len(expected)}')
        for index, (a, b) in enumerate(zip(actual, expected)):
            if tuple(a) != sb.FIELDS:
                self.fail(f'{where}: record {index} has keys in the wrong order or set')
            if a != b:
                fields = [k for k in sb.FIELDS if a[k] != b[k]]
                self.fail(f'{where}: record {index} differs in {fields}')


class LetterTables(NoLeak):
    def test_codes_cover_all_216_combinations_injectively(self):
        middles = {}
        for combination in COMBINATIONS:
            middle = sb.encode_middle(record_of('x', 'y', combination), 'codes')
            self.assertRegex(middle, r'^[shcSHC][pf]?[gc]?[dri]?$')
            self.assertEqual(middle[0].isupper(), combination[1] == 'negative')
            self.assertNotIn(middle, middles, 'two combinations share a codes middle')
            middles[middle] = combination
            style, features = sb.parse_middle(middle)
            self.assertEqual(style, 'codes')
            self.assertEqual(tuple(features[k] for k in sb.FIELDS[2:]), combination)
        self.assertEqual(len(middles), 216)
        self.assertEqual(sb.encode_middle({'subject': 'x', 'object': 'y', 'relation': 'see'}, 'codes'), 's')
        self.assertEqual(sb.encode_middle(record_of('x', 'y', ('help', 'negative', 'future', 'progressive', 'inferred')), 'codes'), 'Hfgi')

    def test_words_cover_all_216_combinations_injectively(self):
        middles = {}
        for combination in COMBINATIONS:
            middle = sb.encode_middle(record_of('x', 'y', combination), 'words')
            parts = middle.split(' ')
            self.assertEqual(parts[0], combination[0])
            self.assertEqual('negative' in parts, combination[1] == 'negative')
            for default in sb.DEFAULTS.values():
                self.assertNotIn(default, parts)
            self.assertNotIn(middle, middles, 'two combinations share a words middle')
            middles[middle] = combination
            style, features = sb.parse_middle(middle)
            self.assertEqual(style, 'words')
            self.assertEqual(tuple(features[k] for k in sb.FIELDS[2:]), combination)
        self.assertEqual(len(middles), 216)
        self.assertEqual(sb.encode_middle({'subject': 'x', 'object': 'y', 'relation': 'see'}, 'words'), 'see')
        self.assertEqual(sb.encode_middle(record_of('x', 'y', ('help', 'negative', 'future', 'progressive', 'inferred')), 'words'),
                         'help negative future progressive inferred')

    def test_styles_are_disjoint(self):
        for combination in COMBINATIONS:
            codes = sb.encode_middle(record_of('x', 'y', combination), 'codes')
            words = sb.encode_middle(record_of('x', 'y', combination), 'words')
            self.assertIsNone(sb._parse_words(codes))
            self.assertIsNone(sb._parse_codes(words))


class RoundTrips(NoLeak):
    def check_batches(self, rng, batches, where):
        for index, records in enumerate(batches):
            expected = [sb._normalize_record(r, i) for i, r in enumerate(records)]
            given = [with_defaults_dropped(r, rng) for r in records]
            for style in sb.STYLES:
                for version, marker in ((1, False), (2, False), (2, True)):
                    text = sb.encode(given, style=style, version=version, marker=marker)
                    label = f'{where} batch {index} {style} v{version}{" marker" if marker else ""}'
                    self.assertTrue(text.endswith('\n') and not text.endswith('\n\n'), label + ': must end with one LF')
                    lines = text.split('\n')[:-1]
                    self.assertEqual(len(lines), len(records) + (1 if version == 1 or marker else 0), label)
                    if version == 1:
                        self.assertEqual(lines[0], f'{sb.HEADER_TAG}\t1\t{style}\t{sb.POLICY}\t0\t{len(records)}', label)
                        self.same_text(text.split('\n', 1)[1], sb.encode(given, style=style, version=2), label + ' vs v2 body')
                    elif marker:
                        self.assertEqual(lines[0], sb.MARKER, label)
                    decoded, info = sb.decode(text)
                    self.same_records(decoded, expected, label)
                    self.assertEqual(info, {'version': version, 'style': style, 'records': len(records), 'marker': marker}, label)
                    self.same_text(sb.encode(decoded, style=style, version=version, marker=marker), text, label + ' re-encode')

    def test_2000_random_alphanumeric_records(self):
        rng = random.Random(20261015)
        batches, total = [], 0
        while total < 2000:
            size = min(rng.randint(1, 20), 2000 - total)
            batches.append(random_batch(rng, size))
            total += size
        self.assertEqual(sum(len(b) for b in batches), 2000)
        self.check_batches(rng, batches, 'alnum')

    def test_escaped_identifiers(self):
        rng = random.Random(7)
        batches = [random_batch(rng, rng.randint(1, 6), ODD) for _ in range(300)]
        batches.append([record_of('if', 'if', COMBINATIONS[0]), record_of('a\u0301', '\u00e9', COMBINATIONS[1])])
        self.check_batches(rng, batches, 'odd')

    def test_escaping_rules(self):
        esc = sb.escape_identifier
        self.assertEqual(esc('plain-id_7.x'), 'plain-id_7.x')
        self.assertEqual(esc('if'), '\\u0069f')
        self.assertEqual(esc('ifx'), 'ifx')
        self.assertEqual(esc('IF'), 'IF')
        self.assertEqual(esc('x@y'), 'x@y')
        self.assertEqual(esc('"a\\b"'), '\\"a\\\\b\\"')
        self.assertEqual(esc('<a[b]>'), '\\u003ca\\u005bb]>')
        self.assertEqual(esc('\x00\x08\x0b\x0c\r\x1f'), '\\u0000\\b\\u000b\\f\\r\\u001f')
        self.assertEqual(esc('\x7f\x80'), '\x7f\x80')  # kept when the identifier is NFC
        self.assertEqual(esc('e\u0301'), 'e\\u0301')  # not NFC: non-ASCII escaped, ASCII kept
        self.assertEqual(esc('\u00e9e\u0301\x7f'), '\\u00e9e\\u0301\\u007f')
        self.assertEqual(esc('\u00e9'), '\u00e9')
        self.assertEqual(esc('\u2126'), '\\u2126')  # NFC singleton
        self.assertEqual(esc('\u1100\u1161'), '\\u1100\\u1161')  # composes to a Hangul syllable
        self.assertEqual(esc('\u0378'), '\\u0378')  # unassigned
        self.assertEqual(esc('a\U0001fffe'), 'a\\ud83f\\udffe')  # noncharacter above the BMP
        self.assertEqual(esc('\U0001f600\u00e9'), '\U0001f600\u00e9')
        # the NFC test is on the light spelling: a mark after an escape whose last letter it
        # composes with selects full mode, one it does not compose with stays light
        self.assertEqual(esc('\r\u0323'), '\\r\\u0323')  # r + U+0323 would compose
        self.assertEqual(esc('\x0e\u0323'), '\\u000e\\u0323')  # e + U+0323 would compose
        self.assertEqual(esc('\x0c\u0301'), '\\f\u0301')  # f + U+0301 does not
        self.assertEqual(esc('"\u0301'), '\\"\u0301')
        self.assertEqual(esc('\u0323'), '\u0323')
        self.assertEqual(esc('ab\u0301'), 'ab\u0301')
        for wire in ['\\u0069f', 'e\\u0301', '\\u2126', '\\"a\\\\b\\"', 'a\\ud83f\\udffe', '\\u003c', '\\r\\u0323', '\\f\u0301']:
            self.assertEqual(esc(sb.unescape_identifier(wire)), wire)
        for wire in ['x\\u0069f', '\\u007f', '\\f\\u0301', '\\r\u0323']:
            with self.assertRaises(ValueError):
                sb.unescape_identifier(wire)

    def test_encode_fills_defaults_and_accepts_kind_pred(self):
        text = sb.encode([{'subject': 'x', 'object': 'y', 'relation': 'see', 'kind': 'pred'}], version=2)
        self.assertEqual(text, 'x\ts\ty\n')
        records, info = sb.decode(text)
        self.assertEqual(records, [{'subject': 'x', 'object': 'y', 'relation': 'see', 'polarity': 'positive',
                                    'tense': 'present', 'aspect': 'simple', 'evidence': 'unspecified'}])
        self.assertEqual(info, {'version': 2, 'style': 'codes', 'records': 1, 'marker': False})

    def test_cli_round_trip(self):
        import json
        import subprocess
        records = [record_of('node7', 'leaf2', ('help', 'negative', 'future', 'simple', 'direct')),
                   record_of('q', 'w', ('see', 'positive', 'present', 'simple', 'unspecified'))]
        jsonl = ''.join(json.dumps(r) + '\n' for r in records).encode()
        script = os.path.join(HERE, 'sylang_batch.py')

        def run(args, data):
            done = subprocess.run([sys.executable, '-I', script, *args], input=data, capture_output=True)
            self.assertEqual(done.returncode, 0, done.stderr.decode(errors='replace'))
            return done.stdout

        v2 = run(['encode', '--version', '2', '--style', 'words'], jsonl)
        self.assertEqual(v2, sb.encode(records, style='words', version=2).encode())
        v1 = run(['convert', '--to', '1'], v2)
        self.assertEqual(v1, sb.encode(records, style='words', version=1).encode())
        self.assertEqual(run(['convert', '--to', '2', '--marker'], v1), sb.encode(records, style='words', version=2, marker=True).encode())
        back = [json.loads(line) for line in run(['decode'], v1).decode().splitlines()]
        self.assertEqual(back, records)
        done = subprocess.run([sys.executable, '-I', script, 'decode'], input=b'x s y\n', capture_output=True)
        self.assertEqual(done.returncode, 2)
        self.assertEqual(done.stdout, b'')


class Malformed(NoLeak):
    HEADER = f'{sb.HEADER_TAG}\t1\tcodes\t{sb.POLICY}\t0\t'

    def test_decode_rejects(self):
        H = self.HEADER
        cases = [
            ('', ValueError, 'empty'),
            ('\n', ValueError, 'blank line'),
            ('x\ts\ty', ValueError, 'no trailing LF'),
            ('x\ts\ty\n\n', ValueError, 'blank trailing line'),
            ('x s y\n', ValueError, 'missing tabs'),
            ('x\ts\n', ValueError, 'two fields'),
            ('x\ts\ty\tz\n', ValueError, 'four fields'),
            ('x\tz\ty\n', ValueError, 'unknown letter'),
            ('x\tsz\ty\n', ValueError, 'unknown feature letter'),
            ('x\tsgp\ty\n', ValueError, 'letters in the wrong order'),
            ('x\tspp\ty\n', ValueError, 'repeated feature letter'),
            ('x\t\ty\n', ValueError, 'empty middle'),
            ('x\tsee  past\ty\n', ValueError, 'double space'),
            ('x\tsee positive\ty\n', ValueError, 'default spelled out'),
            ('x\tsee past past\ty\n', ValueError, 'repeated word'),
            ('x\tsee direct past\ty\n', ValueError, 'words in the wrong order'),
            ('x\tpast see\ty\n', ValueError, 'relation not first'),
            ('x\tnegative\ty\n', ValueError, 'no relation'),
            ('x\tsee\ty\nx\ts\ty\n', ValueError, 'mixed styles'),
            (H + '2\nx\ts\ty\n', ValueError, 'R too large'),
            (H + '1\nx\ts\ty\nx\ts\ty\n', ValueError, 'R too small'),
            (H + '0\n', ValueError, 'R is 0'),
            (H + 'two\nx\ts\ty\n', ValueError, 'R not a number'),
            (f'{sb.HEADER_TAG}\t1\tcodes\t{sb.POLICY}\t1\t1\nx\ts\ty\n', NotImplementedError, 'interning N>0'),
            (f'{sb.HEADER_TAG}\t1\tcodes\t{sb.POLICY}\t3\t1\n@0\ts\ty\n', NotImplementedError, 'interning N>0 with reference'),
            (f'{sb.HEADER_TAG}\t1\tcodes\tfull\t0\t1\nx\ts\ty\n', NotImplementedError, 'non-selective policy'),
            (f'{sb.HEADER_TAG}\t2\tcodes\t{sb.POLICY}\t0\t1\nx\ts\ty\n', ValueError, 'header version 2'),
            (f'{sb.HEADER_TAG}\t1\tjson\t{sb.POLICY}\t0\t1\nx\ts\ty\n', ValueError, 'unknown style'),
            (f'{sb.HEADER_TAG}\t1\twords\t{sb.POLICY}\t0\t1\nx\ts\ty\n', ValueError, 'header style mismatch'),
            (f'{sb.HEADER_TAG}\t1\tcodes\t{sb.POLICY}\t0\n', ValueError, 'short header'),
            (f'{sb.HEADER_TAG}\t1\tcodes\t{sb.POLICY}\t0\t1\t0\nx\ts\ty\n', ValueError, 'long header'),
            (f'{sb.HEADER_TAG}\n', ValueError, 'bare tag is a one-field record'),
            (sb.MARKER + '\n', ValueError, 'marker only'),
            (sb.MARKER + '\n' + sb.MARKER + '\nx\ts\ty\n', ValueError, 'double marker'),
            ('\ts\ty\n', ValueError, 'empty subject'),
            ('x\ts\t\n', ValueError, 'empty object'),
            ('@x\ts\ty\n', ValueError, 'raw leading @'),
            ('\\u0040x\ts\ty\n', ValueError, 'escaped leading @'),
            ('if\ts\ty\n', NotImplementedError, 'if record'),
            ('x\\u0061\ts\ty\n', ValueError, 'non-canonical escape'),
            ('x"\ts\ty\n', ValueError, 'raw quote'),
            ('x<\ts\ty\n', ValueError, 'raw <'),
            ('x[\ts\ty\n', ValueError, 'raw ['),
            ('x\u0378\ts\ty\n', ValueError, 'raw unassigned'),
            ('e\u0301\ts\ty\n', ValueError, 'raw non-NFC'),
            ('x\\\ts\ty\n', ValueError, 'dangling backslash'),
            ('x\\q\ts\ty\n', ValueError, 'unknown escape'),
            ('x\\u12\ts\ty\n', ValueError, 'short hex'),
            ('x\\u12G4\ts\ty\n', ValueError, 'bad hex'),
            ('x\\U0061\ts\ty\n', ValueError, 'upper-case U'),
            ('x\\ud83f\ts\ty\n', ValueError, 'lone high surrogate escape'),
            ('x\\udffe\ts\ty\n', ValueError, 'lone low surrogate escape'),
            ('x\\u00e9\ts\ty\n', ValueError, 'needless escape of an NFC letter'),
            ('x\\t\ts\ty\n', ValueError, 'escaped tab'),
            ('x\\n\ts\ty\n', ValueError, 'escaped LF'),
            ('x\ts\ty\r\n', ValueError, 'CRLF'),
            ('x\x7f\ts\ty\n\\u007f\ts\ty\n', ValueError, 'DEL escaped in an NFC identifier'),
            (H.replace('\t0\t', '\t0\t0') + '1\nx\ts\ty\n', ValueError, 'R with a leading zero'),
            (H.replace('\t0\t', '\t00\t') + '1\nx\ts\ty\n', ValueError, 'N with a leading zero'),
            (H + '\uff11\nx\ts\ty\n', ValueError, 'fullwidth digit in R'),
            (H + '\u0661\nx\ts\ty\n', ValueError, 'Arabic-Indic digit in R'),
            ('x\ts\tif\n', ValueError, 'object spelled if (non-canonical, not an if record)'),
            (H.lower() + '1\nx\ts\ty\n', ValueError, 'damaged header: lower-case tag'),
            ('\ufeff' + H + '1\nx\ts\ty\n', ValueError, 'damaged header: BOM before the tag'),
            ('a\tb\tc\td\te\tf\nx\ts\ty\n', ValueError, 'damaged header: six fields'),
            ('x\ts\ty\n' * 1001, ValueError, 'more than 1000 records'),
            ('x' * 257 + '\ts\ty\n', ValueError, 'identifier of 257 scalars'),
        ]
        for text, exception, label in cases:
            with self.subTest(label):
                with self.assertRaises(exception, msg=label):
                    sb.decode(text)
        with self.assertRaises(TypeError):
            sb.decode(b'x\ts\ty\n')

    def test_valid_edge_cases_decode(self):
        ok = ['x\ts\ty\n', '#x\ts\ty\n', 'x@y\tS\ty\n', 'x\ts\t#sylang2\n', 'x\ts\t' + sb.HEADER_TAG + '\n',
              'x\x7f\ts\ty\n', 'x ]>}\ts\ty\n', 'xif\ts\ty\n', '\\u0069f\ts\t\\u0069f\n', 'x\tcc\ty\n']
        for text in ok:
            records, info = sb.decode(text)
            self.assertEqual(info['records'], 1)
            self.same_text(sb.encode(records, style=info['style'], version=2), text, 'edge case re-encode')
        # A v2 text whose first subject is the header tag is ambiguous without the marker (it reads
        # as a malformed v1 header); with the marker, or under a v1 header, it decodes.
        with self.assertRaises(ValueError):
            sb.decode(f'{sb.HEADER_TAG}\ts\ty\n')
        for text in [sb.MARKER + f'\n{sb.HEADER_TAG}\ts\ty\n', self.HEADER + f'1\n{sb.HEADER_TAG}\ts\ty\n']:
            records, info = sb.decode(text)
            self.assertEqual(records[0]['subject'], sb.HEADER_TAG)
            self.same_text(sb.encode(records, version=info['version'], marker=info['marker']), text, 'reserved subject')

    def test_encode_rejects(self):
        good = {'subject': 'x', 'object': 'y', 'relation': 'see'}
        cases = [
            ([], ValueError, 'empty batch'),
            ([{**good, 'subject': ''}], ValueError, 'empty identifier'),
            ([{**good, 'subject': 'a\tb'}], ValueError, 'tab'),
            ([{**good, 'object': 'a\nb'}], ValueError, 'LF'),
            ([{**good, 'subject': '@x'}], ValueError, 'leading @'),
            ([{**good, 'subject': '\ud800'}], ValueError, 'lone surrogate'),
            ([{**good, 'subject': 7}], TypeError, 'non-str identifier'),
            ([{**good, 'relation': 'look'}], ValueError, 'unknown relation'),
            ([{**good, 'polarity': 'pos'}], ValueError, 'unknown polarity'),
            ([{**good, 'tense': 'now'}], ValueError, 'unknown tense'),
            ([{**good, 'aspect': 'done'}], ValueError, 'unknown aspect'),
            ([{**good, 'evidence': 'seen'}], ValueError, 'unknown evidence'),
            ([{**good, 'extra': 1}], ValueError, 'unknown key'),
            ([{'subject': 'x', 'object': 'y'}], ValueError, 'missing relation'),
            ([{'subject': 'x', 'relation': 'see'}], ValueError, 'missing object'),
            ([{**good, 'kind': 'if'}], NotImplementedError, 'if record'),
            (['x\ts\ty'], TypeError, 'record not a dict'),
            ([good] * 1001, ValueError, 'more than 1000 records (the reference cap)'),
            ([{**good, 'subject': 'x' * 257}], ValueError, 'identifier of 257 scalars (the reference cap)'),
            ([{**good, 'object': 'e\u0301' * 129}], ValueError, '258 scalars in decomposed form'),
        ]
        for records, exception, label in cases:
            for version in sb.VERSIONS:
                with self.subTest(label, version=version):
                    with self.assertRaises(exception, msg=label):
                        sb.encode(records, version=version)
        with self.assertRaises(ValueError):
            sb.encode([good], style='json')
        with self.assertRaises(ValueError):
            sb.encode([good], version=3)
        with self.assertRaises(ValueError):
            sb.encode([good], version=1, marker=True)
        with self.assertRaises(ValueError):
            sb.encode([{**good, 'subject': sb.HEADER_TAG}], version=2)
        sb.encode([{**good, 'subject': sb.HEADER_TAG}], version=1)  # fine under a header
        sb.encode([good, {**good, 'subject': sb.HEADER_TAG}], version=2)  # only the first subject is reserved
        sb.encode([good] * 1000, version=1)  # exactly the cap
        sb.encode([{**good, 'subject': 'x' * 256}], version=2)
        text = sb.encode([good] * 1000, version=1)
        self.assertEqual(sb.decode(text)[1]['records'], 1000)


@unittest.skipUnless(os.environ.get('SYLANG_PACKET'), 'set SYLANG_PACKET=<packet directory> to run the parity test')
class PacketParity(NoLeak):
    """Byte-for-byte parity with the reference v1 renderer, called only through the black-box wrapper."""

    @classmethod
    def setUpClass(cls):
        import language as L
        from followup import Runtime
        cls.L = L
        cls.renderer = L.Renderer(Runtime(os.environ['SYLANG_PACKET']))

    def statements(self, records):
        return [self.L._statement(r['subject'], r['object'], r['relation'], r['polarity'], r['tense'], r['aspect'], r['evidence'])
                for r in records]

    def check(self, records, where):
        for style in sb.STYLES:
            reference = self.renderer.payload(style, self.statements(records))
            self.same_text(sb.encode(records, version=1, style=style), reference, f'{where} {style}')
            decoded, info = sb.decode(reference)
            self.same_records(decoded, [sb._normalize_record(r, i) for i, r in enumerate(records)], f'{where} {style} decode')
            self.assertEqual(info, {'version': 1, 'style': style, 'records': len(records), 'marker': False}, where)

    def test_letter_tables(self):
        records = [record_of('x', 'y', c) for c in COMBINATIONS]
        self.check(records, 'all 216 combinations')

    def test_3000_random_batches(self):
        rng = random.Random(3000)
        for index in range(3000):
            self.check(random_batch(rng, rng.randint(1, 20)), f'batch {index}')

    def test_escaping_parity(self):
        rng = random.Random(11)
        for index in range(300):
            self.check(random_batch(rng, rng.randint(1, 8), ODD), f'odd batch {index}')
        self.check([record_of('if', 'if', COMBINATIONS[5])], 'if identifier')

    def test_every_bmp_code_point(self):
        """Each BMP code point (except tab, LF and surrogates) in a later and in the first position."""
        points = [chr(c) for c in range(0x10000) if not 0xd800 <= c <= 0xdfff and chr(c) not in '\t\n']
        for start in range(0, len(points), 1000):
            chunk = points[start:start + 1000]
            self.check([record_of('a' + ch, 'b', COMBINATIONS[0]) for ch in chunk], f'BMP after a letter from {start}')
            self.check([record_of(ch, 'b', COMBINATIONS[0]) for ch in chunk if ch != '@'], f'BMP leading from {start}')

    def test_astral_sample(self):
        points = [chr(c) for c in range(0x10000, 0x110000, 97)]
        for start in range(0, len(points), 1000):
            chunk = points[start:start + 1000]
            self.check([record_of('a' + ch, ch, COMBINATIONS[0]) for ch in chunk], f'astral from {start}')


if __name__ == '__main__':
    unittest.main()

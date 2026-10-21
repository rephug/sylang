"""Round-trip and comparison-integrity checks for diffusion surface units."""

from __future__ import annotations

import importlib.util
import unittest

from evaluation.diffusion_surfaces import (
    DEFAULT_TOKENIZER_MANIFEST, SURFACE_FORMATS, decode_surface_tokens,
    surface_text, surface_tokens, tokenize_surface, tokenizer_metadata,
)
from evaluation.harness import EvaluationError, load_suite, read_json
from sylang_core import decode, encode


def _qwen_available():
    if importlib.util.find_spec("tokenizers") is None:
        return False
    manifest = read_json(DEFAULT_TOKENIZER_MANIFEST.read_text(encoding="utf-8"))
    return all((DEFAULT_TOKENIZER_MANIFEST.parent / entry["tokenizer_path"]).exists()
               for entry in manifest["tokenizers"])


class DiffusionSurfaceTests(unittest.TestCase):
    def test_surrogates_preserve_existing_gold_semantics_and_exact_text(self):
        rows, _ = load_suite()
        for row in rows:
            for format_name in SURFACE_FORMATS:
                for tokenizer in ("lexical", "utf8_bytes"):
                    with self.subTest(fixture=row["id"], format=format_name, tokenizer=tokenizer):
                        tokens = surface_tokens(row["ast"], format_name, tokenizer)
                        text = decode_surface_tokens(tokens, tokenizer)
                        self.assertEqual(text, encode(row["ast"], format_name))
                        self.assertEqual(decode(text, format_name), row["ast"])

    def test_lexical_units_preserve_whitespace_and_split_quoted_literals(self):
        text = '"Ada Lovelace"\t+\n"line\\nquote\\\""  é\u0301🙂'
        tokens = tokenize_surface(text)
        self.assertIn("Ada", tokens)
        self.assertIn("Lovelace", tokens)
        self.assertNotIn('"Ada Lovelace"', tokens)
        self.assertIn("\t", tokens)
        self.assertIn("  ", tokens)
        self.assertEqual(decode_surface_tokens(tokens), text)

    def test_surrogate_metadata_does_not_claim_trained_model_tokens(self):
        for tokenizer in ("lexical", "utf8_bytes"):
            metadata = tokenizer_metadata(tokenizer)
            self.assertEqual(metadata["kind"], "surrogate")
            self.assertFalse(metadata["native_model_tokens"])
            self.assertFalse(metadata["quoted_literals_atomic"])

    def test_bad_units_and_unsupported_formats_fail(self):
        for tokens, tokenizer in (((12,), "lexical"), ((True,), "utf8_bytes"),
                                   ((256,), "utf8_bytes"), ((-1,), "utf8_bytes"),
                                   ((0xFF,), "utf8_bytes")):
            with self.subTest(tokens=tokens, tokenizer=tokenizer):
                with self.assertRaises(EvaluationError):
                    decode_surface_tokens(tokens, tokenizer)
        with self.assertRaises(EvaluationError):
            surface_text({}, "invented")
        with self.assertRaises(TypeError):
            tokenize_surface(b"not text")

    def test_empty_token_streams_are_reversible(self):
        for tokenizer in ("lexical", "utf8_bytes"):
            self.assertEqual(tokenize_surface("", tokenizer), ())
            self.assertEqual(decode_surface_tokens((), tokenizer), "")

    @unittest.skipUnless(_qwen_available(), "Pinned optional tokenizer runtime/assets unavailable")
    def test_pinned_qwen_tokens_are_real_lossless_surface_units(self):
        rows, _ = load_suite()
        for tokenizer in ("qwen3", "qwen3.5"):
            metadata = tokenizer_metadata(tokenizer)
            self.assertTrue(metadata["native_model_tokens"])
            self.assertFalse(metadata["model_inference_run"])
            self.assertEqual(len(metadata["revision"]), 40)
            self.assertEqual(len(metadata["sha256"]), 64)
            for row in rows:
                for format_name in SURFACE_FORMATS:
                    with self.subTest(fixture=row["id"], format=format_name, tokenizer=tokenizer):
                        # The reviewed Qwen configurations normalize NFC and
                        # recognize reserved markers. Fail closed rather than
                        # silently changing semantic identity or token units.
                        if row["id"] in ("literal-decomposed", "literal-special-token"):
                            with self.assertRaises(EvaluationError):
                                surface_tokens(row["ast"], format_name, tokenizer)
                            continue
                        tokens = surface_tokens(row["ast"], format_name, tokenizer)
                        self.assertTrue(all(type(unit) is int for unit in tokens))
                        self.assertEqual(decode_surface_tokens(tokens, tokenizer),
                                         surface_text(row["ast"], format_name))
            with self.assertRaises(EvaluationError):
                decode_surface_tokens((10**12,), tokenizer)


if __name__ == "__main__":
    unittest.main()

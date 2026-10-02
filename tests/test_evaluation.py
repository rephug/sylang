"""Evaluation checks emphasize denominator correctness and data integrity."""

from __future__ import annotations

import copy
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from evaluation.harness import (DEFAULT_SUITE, EvaluationError, LocalTokenizer,
    comprehension_tasks, evaluate, jsonl_lines, load_suite, prompt_for,
    ratio_summary, read_json, score_answers, sha256)
from sylang_core import FORMATS, decode, encode


class EvaluationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def suite_copy(self):
        path = self.root / DEFAULT_SUITE.name
        path.write_bytes(DEFAULT_SUITE.read_bytes())
        manifest = self.root / "manifest.json"
        manifest.write_bytes(DEFAULT_SUITE.with_name("manifest.json").read_bytes())
        return path, manifest

    def write_answers(self, answers):
        path = self.root / "answers.jsonl"
        path.write_text("".join(json.dumps(answer, ensure_ascii=False) + "\n"
                                for answer in answers), encoding="utf-8")
        return path

    def test_gold_fixture_content_roundtrips_without_unicode_normalization(self):
        rows, _ = load_suite()
        self.assertGreaterEqual(len(rows), 30)
        by_id = {row["id"]: row for row in rows}
        self.assertNotEqual(by_id["literal-composed"]["ast"], by_id["literal-decomposed"]["ast"])
        self.assertIn("\u2028", by_id["literal-unicode-separators"]["ast"]["statement"]["object"])
        for row in rows:
            for format_name in FORMATS:
                with self.subTest(fixture=row["id"], format=format_name):
                    self.assertEqual(row["ast"], decode(encode(row["ast"], format_name), format_name))

    def test_rejects_tampered_fixture_bytes(self):
        path, manifest = self.suite_copy()
        path.write_bytes(path.read_bytes() + b" ")
        with self.assertRaisesRegex(EvaluationError, "manifest mismatch"):
            load_suite(path, manifest)

    def test_rejects_duplicate_fixture_ids_even_with_matching_hash(self):
        path, manifest_path = self.suite_copy()
        lines = jsonl_lines(path.read_text(encoding="utf-8"))
        data = ("\n".join(lines + [lines[0]]) + "\n").encode("utf-8")
        path.write_bytes(data)
        manifest = read_json(manifest_path.read_text(encoding="utf-8"))
        manifest.update(sha256=sha256(data), bytes=len(data), count=len(lines) + 1)
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
        with self.assertRaisesRegex(EvaluationError, "duplicate fixture"):
            load_suite(path, manifest_path)

    def test_jsonl_preserves_unicode_line_separators(self):
        self.assertEqual(jsonl_lines('"a\u0085b\u2028c\u2029d"\n'), ['"a\u0085b\u2028c\u2029d"'])
        with self.assertRaises(EvaluationError):
            read_json('{"x":1,"x":2}')
        with self.assertRaises(EvaluationError):
            read_json('{"x":NaN}')

    def test_macro_and_micro_have_correct_distinct_denominators(self):
        result = ratio_summary([(1, 2), (9, 90)])
        self.assertAlmostEqual(result["macro_ratio"], 0.3)
        self.assertAlmostEqual(result["micro_ratio"], 10 / 92)
        self.assertEqual(result["worst_ratio"], 0.5)
        self.assertEqual(result["numerator_total"], 10)
        self.assertEqual(result["denominator_total"], 92)

    def test_offline_measurement_needs_no_optional_tokenizer(self):
        result = evaluate(repetitions=1)
        self.assertEqual(result["tokenizers"], [])
        self.assertEqual(len(result["rows"]), result["fixture_manifest"]["count"] * len(FORMATS))
        self.assertTrue(all(row["semantic_roundtrip_exact"] for row in result["rows"]))
        self.assertEqual(set(result["summary_by_split"]), {"development", "holdout"})
        self.assertFalse(result["methodology"]["inference_run"])

    def test_complete_prompt_is_measured_as_one_string(self):
        seen = []

        class FakeTokenizer:
            metadata = {"id": "test-only"}

            def measure(self, text):
                seen.append(text)
                return dict(tokens=len(text), decoded_text_exact=True,
                            unknown_token_ids=[], special_token_ids=[], tokenize_cpu_ns=0)

        with patch("evaluation.harness.load_tokenizers", return_value=[FakeTokenizer()]):
            result = evaluate(repetitions=1)
        fixtures, _ = load_suite()
        expected_payload = encode(fixtures[0]["ast"], FORMATS[0])
        self.assertEqual(seen[0], expected_payload)
        self.assertEqual(seen[1], prompt_for(expected_payload, FORMATS[0]))
        self.assertIn("Legend:", seen[1])
        self.assertIn("Output schema:", seen[1])
        self.assertIn("dsl", result["summary"]["m"]["tokenizers"]["test-only"]["payload"]["ratios_to"])

    def test_comprehension_scores_missing_responses_as_incomplete(self):
        tasks, expected = comprehension_tasks()
        good = dict(task_id=tasks[0]["task_id"], answer=expected[tasks[0]["task_id"]])
        wrong_ast = copy.deepcopy(expected[tasks[1]["task_id"]])
        wrong_ast["statement"]["polarity"] = "negative"
        wrong = dict(task_id=tasks[1]["task_id"], answer=wrong_ast)
        result = score_answers(self.write_answers([good, wrong]))
        self.assertEqual(result["submitted"], 2)
        self.assertEqual(result["missing"], len(tasks) - 2)
        self.assertEqual(result["exact_semantic_matches"], 1)

    def test_comprehension_preserves_unicode_answers(self):
        tasks, expected = comprehension_tasks()
        task = next(task for task in tasks if task["fixture_id"] == "literal-unicode-separators")
        result = score_answers(self.write_answers([dict(task_id=task["task_id"], answer=expected[task["task_id"]])]))
        self.assertEqual(result["exact_semantic_matches"], 1)

    def test_rejects_duplicate_or_unknown_answer_ids(self):
        tasks, expected = comprehension_tasks()
        row = dict(task_id=tasks[0]["task_id"], answer=expected[tasks[0]["task_id"]])
        with self.assertRaisesRegex(EvaluationError, "duplicate task ID"):
            score_answers(self.write_answers([row, row]))
        with self.assertRaisesRegex(EvaluationError, "Unknown"):
            score_answers(self.write_answers([dict(task_id="unrelated-task", answer={})]))

    def test_invalid_schema_answer_never_counts_as_correct(self):
        tasks, _ = comprehension_tasks()
        result = score_answers(self.write_answers([dict(task_id=tasks[0]["task_id"], answer={})]))
        self.assertEqual(result["exact_semantic_matches"], 0)
        self.assertFalse(result["rows"][0]["schema_valid"])

    @unittest.skipUnless(importlib.util.find_spec("tokenizers"), "Optional pinned tokenizers not installed")
    def test_local_tokenizer_normalization_and_special_markers_are_visible(self):
        from tokenizers import Tokenizer, models, normalizers
        tokenizer = Tokenizer(models.WordLevel({"[UNK]": 0, "é": 1, "<reserved>": 2}, unk_token="[UNK]"))
        tokenizer.normalizer = normalizers.NFC()
        tokenizer.add_special_tokens(["<reserved>"])
        asset = self.root / "tokenizer.json"
        tokenizer.save(str(asset))
        metadata = dict(id="local-test", revision="0" * 40, tokenizer_path=asset.name,
                        sha256=sha256(asset.read_bytes()), license="MIT", source="synthetic-unit-test")
        local = LocalTokenizer(metadata, self.root)
        self.assertFalse(local.measure("e\u0301")["decoded_text_exact"])
        self.assertTrue(local.measure("not-in-vocabulary")["unknown_token_ids"])
        self.assertTrue(local.measure("<reserved>")["special_token_ids"])
        metadata["sha256"] = "0" * 64
        with self.assertRaisesRegex(EvaluationError, "SHA-256 mismatch"):
            LocalTokenizer(metadata, self.root)


if __name__ == "__main__":
    unittest.main()

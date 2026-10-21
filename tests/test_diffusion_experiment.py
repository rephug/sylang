"""Finite-world invariants and analytic controls for the experiment runner."""

from __future__ import annotations

import itertools
import math
import unittest

from evaluation.diffusion_experiment import (
    FIELDS, METRICS, NATIVE_GROUPS, PAD, average_metrics, build_worlds,
    exact_unit_masking, length_information, refinement_schedules, round_groups,
    semantic_record, surface_masking,
)
from evaluation.diffusion_oracle import FiniteOracle
from evaluation.diffusion_surfaces import surface_tokens
from sylang_core import decode, encode, validate


class DiffusionExperimentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.worlds = build_worlds()

    def test_worlds_are_unique_valid_ast_supports_with_declared_sizes(self):
        self.assertEqual(set(self.worlds), {"independent_864", "correlated_18"})
        supports = {}
        for name, expected in (("independent_864", 864), ("correlated_18", 18)):
            documents = self.worlds[name]
            self.assertEqual(len(documents), expected)
            records = {semantic_record(document) for document in documents}
            self.assertEqual(len(records), expected)
            supports[name] = records
            for document in documents:
                validate(document)
                for format_name in ("english", "json", "m"):
                    self.assertEqual(decode(encode(document, format_name), format_name), document)
        self.assertLess(supports["correlated_18"], supports["independent_864"])
        independent = supports["independent_864"]
        domains = [set(record[index] for record in independent) for index in range(len(FIELDS))]
        self.assertEqual([len(domain) for domain in domains], [2, 3, 2, 2, 3, 3, 4])
        self.assertEqual(independent, set(itertools.product(*domains)))
        for record in supports["correlated_18"]:
            self.assertNotEqual(record[0], record[2])
            self.assertEqual(("past", "present", "future").index(record[4]),
                             ("simple", "progressive", "completed").index(record[5]))

    def test_equal_unit_erasure_preserves_p_times_entropy_for_independent_fusion(self):
        records = [semantic_record(document) for document in self.worlds["independent_864"]]
        prior_entropy = math.log2(len(records))
        for name, groups in NATIVE_GROUPS.items():
            with self.subTest(encoding=name):
                # Groups cover every field once, so this remains an injective
                # regrouping of independent semantic factors.
                self.assertEqual(sorted(index for group in groups for index in group),
                                 list(range(len(FIELDS))))
                streams = [tuple(tuple(record[index] for index in group) for group in groups)
                           for record in records]
                oracle = FiniteOracle(streams, records)
                for row in exact_unit_masking(oracle, (0.0, 0.25, 0.5, 1.0)):
                    self.assertAlmostEqual(row["entropy_bits"],
                                           row["mask_probability"] * prior_entropy, places=10)
                    self.assertAlmostEqual(row["conditional_total_correlation_bits"], 0.0, places=10)

    def test_matched_semantic_masks_are_invariant_across_actual_surface_formats(self):
        documents = self.worlds["correlated_18"]
        records = [semantic_record(document) for document in documents]
        baseline = exact_unit_masking(FiniteOracle(records, records), (0.5,))[0]
        # Independently enumerate source states and semantic field masks. The
        # revealed observation is field/value pairs, never surface positions or
        # surface lengths; each actual format uses the same candidate worlds.
        for format_name in ("english", "json", "m"):
            oracle = FiniteOracle([surface_tokens(document, format_name)
                                   for document in documents], records)
            observations = []
            for hidden in itertools.product((False, True), repeat=len(FIELDS)):
                visible = tuple(index for index, erased in enumerate(hidden) if not erased)
                for record in records:
                    candidates = sum(1 << index for index, candidate in enumerate(records)
                                     if all(candidate[field] == record[field] for field in visible))
                    observations.append((1 / (2 ** len(FIELDS) * len(records)),
                                         oracle.posterior_stats(candidates)))
            actual = average_metrics(observations)
            for metric in METRICS:
                with self.subTest(format=format_name, metric=metric):
                    self.assertAlmostEqual(actual[metric], baseline[metric], places=10)

    def test_surface_endpoints_expose_length_information_and_padding_removes_it(self):
        streams = [("A",), ("B",), ("C", "C")]
        records = [(0,), (1,), (2,)]
        oracle = FiniteOracle(streams, records)
        clear, erased = surface_masking(oracle, samples_per_state=2, seed=3, rates=(0.0, 1.0))
        self.assertEqual(clear["entropy_bits"], 0)
        self.assertEqual(clear["bayes_recovery"], 1)
        self.assertEqual(clear["unique_recovery"], 1)
        self.assertAlmostEqual(erased["entropy_bits"], 2 / 3)
        self.assertAlmostEqual(erased["bayes_recovery"], 2 / 3)
        self.assertAlmostEqual(erased["unique_recovery"], 1 / 3)
        self.assertEqual(erased["expectation"], "exact_enumeration")
        self.assertTrue(all(error == 0 for error in erased["mask_standard_error"].values()))
        information = length_information(oracle)
        self.assertAlmostEqual(information["length_only_information_bits"], math.log2(3) - 2 / 3)
        self.assertAlmostEqual(information["unknown_length_full_erasure"]["bayes_recovery"], 1 / 3)
        padded = FiniteOracle([stream + (PAD,) * (2 - len(stream)) for stream in streams], records)
        padded_erasure = surface_masking(padded, 2, 3, rates=(1.0,))[0]
        self.assertAlmostEqual(padded_erasure["entropy_bits"], math.log2(3))
        self.assertAlmostEqual(padded_erasure["bayes_recovery"], 1 / 3)
        self.assertEqual(padded_erasure["unique_recovery"], 0)
        self.assertEqual(length_information(padded)["length_only_information_bits"], 0)

    def test_surface_mask_sampling_is_reproducible_and_reports_mask_uncertainty(self):
        records = [(0, 0), (0, 1), (1, 0), (1, 1)]
        oracle = FiniteOracle(records, records)
        result = surface_masking(oracle, 12, 101, rates=(0.5,))
        self.assertEqual(result, surface_masking(oracle, 12, 101, rates=(0.5,)))
        self.assertEqual(result[0]["expectation"], "stratified_mask_monte_carlo")
        self.assertGreater(result[0]["mask_standard_error"]["entropy_bits"], 0)

    def test_round_groups_partition_the_canvas_for_uneven_and_oversized_budgets(self):
        for length, budget, order in itertools.product((1, 2, 5, 10), (1, 2, 3, 20),
                                                      ("contiguous", "interleaved")):
            with self.subTest(length=length, budget=budget, order=order):
                groups = round_groups(length, budget, order)
                self.assertEqual(len(groups), min(length, budget))
                self.assertTrue(all(groups))
                self.assertEqual(sorted(index for group in groups for index in group), list(range(length)))
        self.assertEqual(round_groups(5, 2, "contiguous"), [[0, 1], [2, 3, 4]])
        self.assertEqual(round_groups(5, 2, "interleaved"), [[0, 2, 4], [1, 3]])
        for length, budget, order in ((0, 1, "contiguous"), (1, 0, "interleaved"),
                                      (-2, 1, "contiguous"), (1, 1, "unknown"),
                                      (True, 1, "contiguous"), (1, True, "contiguous"),
                                      (1.5, 1, "contiguous"), (1, 1.5, "interleaved")):
            with self.subTest(length=length, budget=budget, order=order):
                with self.assertRaises(ValueError):
                    round_groups(length, budget, order)

    def test_refinement_thresholds_measure_distribution_fidelity_not_original_recovery(self):
        # The duplicated bit has two meanings. Independent one-round sampling
        # has only 1/2 valid mass; sequential conditional sampling is exact.
        oracle = FiniteOracle([(0, 0), (1, 1)], [(0,), (1,)])
        result = refinement_schedules(oracle)
        self.assertEqual(result["condition"], "full_erasure_gold_canvas_length_given")
        bucket = result["length_buckets"][0]
        self.assertEqual(bucket["bayes_original_recovery_ceiling"], 0.5)
        self.assertEqual(bucket["exact_joint_sampler_rounds"], 1)
        for order in ("contiguous", "interleaved"):
            schedule = bucket["schedules"][order]
            self.assertEqual(schedule["minimum_tested_rounds_meeting_threshold"], 2)
            first, final = schedule["tested"]
            self.assertAlmostEqual(first["valid_support_mass"], 0.5)
            self.assertAlmostEqual(first["total_variation_to_posterior"], 0.5)
            self.assertAlmostEqual(final["valid_support_mass"], 1)
            self.assertAlmostEqual(final["total_variation_to_posterior"], 0)
            self.assertAlmostEqual(final["original_recovery_probability"], 0.5)

    def test_unit_mask_enumeration_rejects_variable_lengths_and_bad_weights(self):
        with self.assertRaises(ValueError):
            exact_unit_masking(FiniteOracle([("a",), ("b", "b")], [(0,), (1,)]))
        stats = FiniteOracle([(0,), (1,)], [(0,), (1,)]).posterior_stats(3)
        with self.assertRaisesRegex(ValueError, "weights sum"):
            average_metrics(((0.8, stats),))

    def test_masking_rejects_invalid_probabilities_and_inadequate_sample_counts(self):
        oracle = FiniteOracle([(0,), (1,)], [(0,), (1,)])
        for rates in ((), (-0.1,), (1.1,), (float("nan"),),
                      (float("inf"),), (True,), (0.5, 0.5), ("0.5",)):
            with self.subTest(rates=rates):
                with self.assertRaises(ValueError):
                    exact_unit_masking(oracle, rates)
                with self.assertRaises(ValueError):
                    surface_masking(oracle, 2, 7, rates)
        for samples in (0, 1, -1, True, 2.5):
            with self.subTest(samples=samples):
                with self.assertRaises(ValueError):
                    surface_masking(oracle, samples, 7)


if __name__ == "__main__":
    unittest.main()

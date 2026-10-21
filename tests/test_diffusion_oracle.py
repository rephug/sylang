"""Analytically known controls for the finite, uniform infilling oracle."""

import itertools
import math
import unittest

from evaluation.diffusion_oracle import FiniteOracle, geometric_retry_stats


class FinitePosteriorTests(unittest.TestCase):
    def test_independent_bits_have_exact_entropy_and_recovery(self):
        oracle = FiniteOracle(list(itertools.product((0, 1), repeat=2)))
        self.assertEqual(oracle.size, 4)
        stats = oracle.posterior_stats(oracle.full_mask)
        self.assertEqual(stats["entropy_bits"], 2)
        self.assertEqual(stats["bayes_recovery"], 0.25)
        self.assertEqual(stats["field_entropy_bits"], [1, 1])
        self.assertEqual(stats["conditional_total_correlation_bits"], 0)
        posterior = oracle.posterior((0, None), (False, True))
        self.assertEqual(posterior.bit_count(), 2)
        self.assertEqual(oracle.posterior_stats(posterior)["entropy_bits"], 1)

    def test_gold_length_is_an_observation(self):
        oracle = FiniteOracle([(0,), (0, 1), (1, 1)])
        self.assertEqual(oracle.posterior((None,), (True,)), 1)
        self.assertEqual(oracle.posterior((None, None), (True, True)), 6)
        with self.assertRaises(ValueError):
            oracle.posterior((None,), (True,), reveal_length=False)
        padded = FiniteOracle([(0, "PAD"), (0, 1), (1, 1)])
        self.assertEqual(padded.posterior((None, None), (True, True), reveal_length=False), 7)
        self.assertEqual(padded.posterior((None, "PAD"), (True, False), reveal_length=False), 1)

    def test_records_separate_semantic_from_token_entropy(self):
        oracle = FiniteOracle([(0, 0), (1, 1)], [("cold",), ("hot",)])
        stats = oracle.posterior_stats(oracle.full_mask)
        self.assertEqual(stats["field_entropy_bits"], [1])
        self.assertEqual(stats["conditional_total_correlation_bits"], 0)
        stats["field_entropy_bits"][0] = 99
        self.assertEqual(oracle.posterior_stats(oracle.full_mask)["field_entropy_bits"], [1])

    def test_impossible_visible_tokens_and_validation(self):
        oracle = FiniteOracle([(0,), (1,)])
        self.assertEqual(oracle.posterior((2,), (False,)), 0)
        self.assertEqual(oracle.posterior((None,), (True,)), 3)
        for mask in (0, -1, 4, True):
            with self.assertRaises(ValueError):
                oracle.posterior_stats(mask)
        with self.assertRaises(ValueError):
            oracle.posterior((0,), ())
        with self.assertRaises(ValueError):
            oracle.posterior((0,), (1,))
        for streams in ([], [()], [(0,), (0,)], [([0],)]):
            with self.assertRaises(ValueError):
                FiniteOracle(streams)
        with self.assertRaises(ValueError):
            FiniteOracle([(0,), (1,)], [(0,), (0,)])
        with self.assertRaises(ValueError):
            FiniteOracle([(0,), (1,)], [(0,)])


class FiniteCompletionTests(unittest.TestCase):
    def test_independent_fields_sample_exact_posterior_in_one_round(self):
        oracle = FiniteOracle(list(itertools.product((0, 1), repeat=3)))
        result = oracle.completion_stats(oracle.full_mask, [[0, 1, 2]])
        self.assertAlmostEqual(result["valid_support_mass"], 1)
        self.assertAlmostEqual(result["total_variation_to_posterior"], 0)
        self.assertAlmostEqual(result["original_recovery_probability"], 1 / 8)
        self.assertEqual(result["processed_positions"], 3)

    def test_correlated_copies_fail_in_parallel_and_recover_conditionally(self):
        oracle = FiniteOracle([(0, 0), (1, 1)])
        self.assertEqual(oracle.posterior_stats(3)["conditional_total_correlation_bits"], 1)
        parallel = oracle.completion_stats(3, [[0, 1]])
        self.assertAlmostEqual(parallel["valid_support_mass"], 0.5)
        self.assertAlmostEqual(parallel["total_variation_to_posterior"], 0.5)
        self.assertAlmostEqual(parallel["accepted_total_variation_to_posterior"], 0)
        self.assertAlmostEqual(parallel["original_recovery_probability"], 0.25)
        self.assertAlmostEqual(parallel["accepted_original_recovery_probability"], 0.5)
        self.assertAlmostEqual(parallel["expected_retry_rounds"], 2)
        self.assertEqual(parallel["rounds_for_99pct_valid"], 7)
        sequential = oracle.completion_stats(3, [[0], [1]])
        self.assertAlmostEqual(sequential["valid_support_mass"], 1)
        self.assertAlmostEqual(sequential["total_variation_to_posterior"], 0)
        self.assertEqual(sequential["number_rounds"], 2)
        self.assertEqual(sequential["processed_positions"], 4)
        self.assertAlmostEqual(sequential["expected_retry_attempts"], 1)
        self.assertAlmostEqual(sequential["expected_total_prediction_rounds"], 2)
        self.assertEqual(sequential["prediction_rounds_for_99pct_valid"], 2)

    def test_accepted_marginals_can_still_bias_the_posterior(self):
        oracle = FiniteOracle([(0, 0), (0, 1), (1, 0)])
        result = oracle.completion_stats(7, [[0, 1]])
        self.assertAlmostEqual(result["valid_support_mass"], 8 / 9)
        self.assertAlmostEqual(result["total_variation_to_posterior"], 2 / 9)
        self.assertAlmostEqual(result["accepted_total_variation_to_posterior"], 1 / 6)
        # The conditional sampler samples the true joint, even though the
        # posterior's independent marginals assign mass to an invalid pair.
        sequential = oracle.completion_stats(7, [[0], [1]])
        self.assertAlmostEqual(sequential["valid_support_mass"], 1)
        self.assertAlmostEqual(sequential["total_variation_to_posterior"], 0)

    def test_posterior_filtering_is_respected_by_completion(self):
        oracle = FiniteOracle([(0, 0), (0, 1), (1, 1)])
        posterior = oracle.posterior((0, None), (False, True))
        result = oracle.completion_stats(posterior, [[0, 1]])
        self.assertEqual(result["candidate_count"], 2)
        self.assertAlmostEqual(result["valid_support_mass"], 1)
        self.assertAlmostEqual(result["original_recovery_probability"], 0.5)

    def test_invalid_partitions_and_mixed_lengths_fail(self):
        oracle = FiniteOracle([(0, 0), (1, 1)])
        for groups in ([], [[]], [[0]], [[0, 0], [1]], [[0], [2]], [[0], [1.0]], [[True], [0]]):
            with self.assertRaises(ValueError):
                oracle.completion_stats(3, groups)
        with self.assertRaises(ValueError):
            FiniteOracle([(0,), (1, 1)]).completion_stats(3, [[0]])

    def test_many_correlated_copies_use_log_probabilities(self):
        length = 2000
        oracle = FiniteOracle([(0,) * length, (1,) * length])
        result = oracle.completion_stats(3, [list(range(length))])
        self.assertEqual(result["valid_support_mass"], 0)
        self.assertAlmostEqual(result["log_valid_support_mass"], -(length - 1) * math.log(2))
        self.assertEqual(result["total_variation_to_posterior"], 1)
        self.assertAlmostEqual(result["accepted_total_variation_to_posterior"], 0)
        self.assertIsNone(result["expected_retry_rounds"])


class GeometricRetryTests(unittest.TestCase):
    def test_geometric_control_and_boundaries(self):
        self.assertEqual(geometric_retry_stats(1)["rounds_for_confidence"], 1)
        self.assertEqual(geometric_retry_stats(0)["expected_retry_rounds"], None)
        result = geometric_retry_stats(0.5)
        self.assertEqual(result["expected_retry_rounds"], 2)
        self.assertEqual(result["rounds_for_confidence"], 7)
        for probability in (-0.1, 1.1, float("nan"), True):
            with self.assertRaises(ValueError):
                geometric_retry_stats(probability)


class IndependentGroupRefinementTests(unittest.TestCase):
    def test_independent_coordinates_need_one_prediction_round(self):
        oracle = FiniteOracle(list(itertools.product((0, 1), repeat=4)))
        for order in ("forward", "reverse"):
            result = oracle.independent_group_refinement(oracle.full_mask, order)
            self.assertEqual(result["mean_number_rounds"], 1)
            self.assertEqual(result["worst_number_rounds"], 1)
            self.assertEqual(result["mean_processed_positions"], 4)
            self.assertEqual(result["valid_support_mass"], 1)
            self.assertEqual(result["total_variation_to_posterior"], 0)

    def test_correlated_copies_require_next_round_materialization(self):
        # Includes deterministic scaffolding in the initial prediction round.
        oracle = FiniteOracle([("fixed", 0, 0), ("fixed", 1, 1)])
        for order in ("forward", "reverse"):
            result = oracle.independent_group_refinement(oracle.full_mask, order)
            self.assertEqual(result["mean_number_rounds"], 2)
            self.assertEqual(result["worst_number_rounds"], 2)
            self.assertEqual(result["mean_processed_positions"], 6)
            self.assertEqual(result["probability_by_rounds"], [
                {"number_rounds": 2, "meaning_count": 2, "probability": 1.0},
            ])

    def test_pairwise_independence_does_not_imply_joint_independence(self):
        # Every pair of parity bits is independent, but all three cannot be
        # sampled independently without leaving the declared support.
        streams = [(x, y, x ^ y) for x, y in itertools.product((0, 1), repeat=2)]
        oracle = FiniteOracle(streams)
        for order in ("forward", "reverse"):
            result = oracle.independent_group_refinement(oracle.full_mask, order)
            self.assertEqual(result["mean_number_rounds"], 2)
            self.assertEqual(result["valid_support_mass"], 1)
            self.assertEqual(result["total_variation_to_posterior"], 0)

    def test_adaptive_branch_depth_is_exactly_weighted_by_posterior(self):
        streams = [(0, x, y) for x, y in itertools.product((0, 1), repeat=2)]
        streams.extend((1, z, z) for z in range(3))
        oracle = FiniteOracle(streams)
        result = oracle.independent_group_refinement(oracle.full_mask)
        self.assertAlmostEqual(result["mean_number_rounds"], 17 / 7)
        self.assertEqual(result["worst_number_rounds"], 3)
        self.assertEqual(result["probability_by_rounds"], [
            {"number_rounds": 2, "meaning_count": 4, "probability": 4 / 7},
            {"number_rounds": 3, "meaning_count": 3, "probability": 3 / 7},
        ])
        reverse = oracle.independent_group_refinement(oracle.full_mask, "reverse")
        self.assertAlmostEqual(reverse["mean_number_rounds"], 20 / 7)
        self.assertEqual(reverse["valid_support_mass"], 1)
        self.assertGreater(reverse["mean_number_rounds"], result["mean_number_rounds"])

    def test_known_candidate_still_needs_one_materialization_pass(self):
        oracle = FiniteOracle([(0, 0), (1, 1)])
        result = oracle.independent_group_refinement(1)
        self.assertEqual(result["mean_number_rounds"], 1)
        self.assertEqual(result["original_recovery_probability"], 1)

    def test_invalid_order_and_variable_canvas_are_rejected(self):
        with self.assertRaises(ValueError):
            FiniteOracle([(0,), (1,)]).independent_group_refinement(3, "random")
        with self.assertRaises(ValueError):
            FiniteOracle([(0,), (1, 1)]).independent_group_refinement(3)


if __name__ == "__main__":
    unittest.main()

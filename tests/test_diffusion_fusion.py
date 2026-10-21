"""Mathematical checks for finite conditional-sampling experiments."""

from fractions import Fraction
import unittest

from evaluation.diffusion_fusion import (
    ConditionalOracle, WORLDS, distribution_metrics, encoded_prior,
    evaluate_encoding, observation_distribution, run_fusion_experiments,
)


class DiffusionFusionTests(unittest.TestCase):
    def row(self, world_name, encoding, probability=1):
        world = next(world for world in WORLDS if world.name == world_name)
        return evaluate_encoding(world, encoding, Fraction(probability))

    def test_independent_fields_need_one_round_without_invalid_combinations(self):
        row = self.row("independent_bits", "factorized")
        self.assertEqual(row["expected_residual_entropy_bits"], 2)
        self.assertEqual(row["bayes_original_recovery_ceiling"], 0.25)
        self.assertEqual(row["one_round"]["valid_support_mass"], 1)
        self.assertEqual(row["one_round"]["total_variation_to_exact_posterior"], 0)
        self.assertEqual(row["optimal_exact_independent_parallel_rounds"]["worst_case"], 1)

    def test_factorized_correlated_pair_loses_half_support_then_recovers_distribution(self):
        row = self.row("correlated_pair", "factorized")
        self.assertEqual(row["one_round"]["valid_support_mass"], 0.5)
        self.assertEqual(row["one_round"]["total_variation_to_exact_posterior"], 0.5)
        self.assertEqual(row["optimal_exact_independent_parallel_rounds"]["worst_case"], 2)
        self.assertEqual(row["minimum_tested_round_budget_meeting_threshold"], 2)
        final = row["refinement_by_round_budget"][-1]
        self.assertEqual(final["valid_support_mass"], 1)
        self.assertEqual(final["total_variation_to_exact_posterior"], 0)
        # Exact sampling cannot identify the erased original random bit.
        self.assertEqual(row["bayes_original_recovery_ceiling"], 0.5)

    def test_fusion_changes_sampling_dependencies_but_erases_more_information(self):
        factorized = self.row("correlated_pair", "factorized", Fraction(1, 2))
        fused = self.row("correlated_pair", "fused_pairs", Fraction(1, 2))
        self.assertEqual(factorized["expected_residual_entropy_bits"], 0.25)
        self.assertEqual(fused["expected_residual_entropy_bits"], 0.5)
        self.assertEqual(fused["one_round"]["valid_support_mass"], 1)
        self.assertEqual(fused["optimal_exact_independent_parallel_rounds"]["worst_case"], 1)

    def test_redundancy_reduces_erasure_but_full_mask_independent_predictions_disagree(self):
        half = self.row("correlated_pair", "duplicate_factorized", Fraction(1, 2))
        full = self.row("correlated_pair", "duplicate_factorized")
        self.assertEqual(half["expected_residual_entropy_bits"], 1 / 16)
        self.assertEqual(full["one_round"]["valid_support_mass"], 1 / 8)
        self.assertEqual(full["optimal_exact_independent_parallel_rounds"]["worst_case"], 2)

    def test_higher_order_dependency_survives_pairwise_independence(self):
        row = self.row("even_parity", "factorized")
        self.assertEqual(row["one_round"]["valid_support_mass"], 0.5)
        self.assertEqual(row["optimal_exact_independent_parallel_rounds"]["worst_case"], 2)
        fused = self.row("even_parity", "fused_pairs")
        self.assertEqual(fused["one_round"]["valid_support_mass"], 0.5)

    def test_sampling_conditions_on_commitments_and_not_original_gold(self):
        oracle = ConditionalOracle(encoded_prior(WORLDS[1], "factorized"))
        erased = (None, None)
        outputs, rounds = oracle.sample(erased, 2)
        self.assertEqual(outputs, {((0,), (0,)): Fraction(1, 2),
                                  ((1,), (1,)): Fraction(1, 2)})
        self.assertEqual(rounds, 2)
        forced = ((1,), None)
        self.assertEqual(oracle.sample(forced, 1)[0], {((1,), (1,)): Fraction(1)})
        with self.assertRaisesRegex(ValueError, "zero probability"):
            oracle.posterior(((0,), (1,)))

    def test_observation_mass_and_expected_closed_form_recoverability(self):
        prior = encoded_prior(WORLDS[1], "factorized")
        observations = observation_distribution(prior, Fraction(1, 4))
        self.assertEqual(sum(observations.values()), 1)
        row = self.row("correlated_pair", "factorized", Fraction(1, 4))
        self.assertEqual(row["expected_residual_entropy_bits"], 1 / 16)
        self.assertEqual(row["bayes_original_recovery_ceiling"], 31 / 32)
        self.assertEqual(row["unique_recovery_probability"], 15 / 16)
        self.assertEqual(row["one_round"]["valid_support_mass"], 31 / 32)

    def test_average_threshold_does_not_hide_the_worst_mask(self):
        row = self.row("correlated_pair", "duplicate_factorized", Fraction(1, 4))
        self.assertEqual(row["minimum_tested_round_budget_meeting_threshold"], 1)
        self.assertEqual(row["minimum_tested_round_budget_meeting_worst_observation_threshold"], 2)
        self.assertEqual(row["refinement_by_round_budget"][0]["worst_observation_valid_support_mass"], 1 / 8)

    def test_no_mask_has_zero_execution_and_complete_information(self):
        row = self.row("even_parity", "duplicate_factorized", 0)
        self.assertEqual(row["expected_residual_entropy_bits"], 0)
        self.assertEqual(row["bayes_original_recovery_ceiling"], 1)
        self.assertEqual(row["optimal_exact_independent_parallel_rounds"]["worst_case"], 0)
        self.assertEqual(row["refinement_by_round_budget"][0]["expected_executed_rounds"], 0)

    def test_nonuniform_prior_has_exact_conditional_distribution(self):
        prior = {((0,), (0,)): Fraction(4, 5), ((1,), (1,)): Fraction(1, 5)}
        oracle = ConditionalOracle(prior)
        outputs, _ = oracle.sample((None, None), 2)
        self.assertEqual(outputs, prior)
        self.assertEqual(distribution_metrics(outputs, prior)["total_variation_to_exact_posterior"], 0)

    def test_complete_suite_is_deterministic_and_every_exact_plan_preserves_support(self):
        results = run_fusion_experiments()
        self.assertEqual(len(results["observations"]), 60)
        self.assertEqual(results, run_fusion_experiments())
        for row in results["observations"]:
            final = row["refinement_by_round_budget"][-1]
            self.assertAlmostEqual(final["valid_support_mass"], 1)
            self.assertAlmostEqual(final["total_variation_to_exact_posterior"], 0)
            self.assertIsNotNone(row["minimum_tested_round_budget_meeting_threshold"])


if __name__ == "__main__":
    unittest.main()

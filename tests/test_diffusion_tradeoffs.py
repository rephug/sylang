"""Boundary checks for hypothetical work accounting, without model claims."""

import unittest

from tools.analyze_diffusion_tradeoffs import comparison, work_summary


class DiffusionWorkAccounting(unittest.TestCase):
    def test_independent_requests_square_before_summing(self):
        result = work_summary([2, 3])
        self.assertEqual(result["sum_squared_lengths"], 13)
        self.assertNotEqual(result["sum_squared_lengths"], result["tokens"] ** 2)

    def test_fixed_blocks_pad_each_request_and_respect_boundary(self):
        result = work_summary([31, 32, 33], 32)
        self.assertEqual(result["blocks"], 4)
        self.assertEqual(result["padded_slots"], 128)
        self.assertEqual(result["unused_slots"], 32)

    def test_compression_within_block_does_not_imply_fewer_blocks(self):
        result = comparison([17], [22], ["sample"])
        self.assertGreater(result["ideal_full_sequence"]["linear_equal_steps_speedup"], 1)
        self.assertEqual(result["ideal_fixed_block"]["32"]["equal_steps_ideal_speedup"], 1)
        self.assertEqual(result["ideal_fixed_block"]["32"]["equal_blocks_fixtures"], 1)

    def test_step_ratio_threshold_handles_candidate_expansion(self):
        result = comparison([4, 6], [2, 3], ["first", "second"])
        ideal = result["ideal_full_sequence"]
        self.assertEqual(ideal["linear_candidate_to_baseline_steps_break_even"], 0.5)
        self.assertEqual(ideal["quadratic_candidate_to_baseline_steps_break_even"], 0.25)

    def test_mismatched_pairing_and_invalid_lengths_fail(self):
        with self.assertRaises(ValueError):
            comparison([1], [1, 2], ["first"])
        for lengths in ([], [0], [1.5], [True]):
            with self.assertRaises(ValueError):
                work_summary(lengths)


if __name__ == "__main__":
    unittest.main()

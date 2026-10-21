"""Exact finite-state infilling controls; no learned model or model tokenization.

The prior is uniform over the supplied, injectively encoded records. Posterior
filtering is exact. Completion probabilities describe a particular idealized
sampler: within a round it draws positions independently from their exact
marginals, then conditions the following round on those draws. Invalid prefixes
are absorbing failures. A round never reveals information from the hidden gold
record. Additional rounds can improve consistency, not the Bayes ceiling for
identifying an ambiguous original.
"""

from __future__ import annotations

from functools import lru_cache
from collections import Counter
import math
import sys
from typing import Hashable, Sequence


def _members(mask: int):
    while mask:
        bit = mask & -mask
        yield bit.bit_length() - 1
        mask ^= bit


def _logsumexp(values: Sequence[float]) -> float:
    maximum = max(values)
    return maximum + math.log(math.fsum(math.exp(x - maximum) for x in values))


def geometric_retry_stats(
    success_probability: float, confidence: float = 0.99,
) -> dict[str, float | int | None]:
    """Repeated independent attempts with a fixed acceptance probability.

    ``None`` means an infinite or floating-point-unrepresentable expectation.
    This is rejection sampling, not a prediction of neural refinement passes.
    """
    if (isinstance(success_probability, bool)
            or not math.isfinite(success_probability)
            or not 0 <= success_probability <= 1):
        raise ValueError("success_probability must lie in [0, 1]")
    if not math.isfinite(confidence) or not 0 < confidence < 1:
        raise ValueError("confidence must lie strictly between zero and one")
    if success_probability == 0:
        expected, rounds = None, None
    elif success_probability == 1:
        expected, rounds = 1.0, 1
    else:
        expected = 1.0 / success_probability
        if not math.isfinite(expected):
            expected = None
        quotient = math.log1p(-confidence) / math.log1p(-success_probability)
        rounds = math.ceil(quotient) if math.isfinite(quotient) else None
    return {
        "success_probability": success_probability,
        "confidence": confidence,
        "expected_retry_rounds": expected,
        "rounds_for_confidence": rounds,
    }


class FiniteOracle:
    """Uniform oracle over unique, hashable token streams.

    ``semantic_records`` are row-aligned tuples of fixed-arity semantic fields.
    They must be unique: synonyms of the same record are not extra prior mass.
    If omitted, equal-length streams themselves serve as the field tuples;
    mixed-length streams have no field-level entropy statistics.

    Public integer bitsets index rows in ``streams``. ``token_masks[position]``
    and ``masks_by_field[field]`` map a value to its row bitset.
    """

    def __init__(
        self,
        streams: Sequence[tuple[Hashable, ...]],
        semantic_records: Sequence[tuple[Hashable, ...]] | None = None,
    ):
        self.streams = tuple(tuple(stream) for stream in streams)
        if not self.streams or any(not stream for stream in self.streams):
            raise ValueError("at least one nonempty token stream is required")
        try:
            unique = set(self.streams)
        except TypeError as error:
            raise ValueError("tokens must be hashable") from error
        if len(unique) != len(self.streams):
            raise ValueError("streams must be unique (injective encoding)")
        self.size = self.corpus_size = len(self.streams)
        self.full_mask = (1 << self.size) - 1
        self.length_masks: dict[int, int] = {}
        token_masks: list[dict[Hashable, int]] = [
            {} for _ in range(max(map(len, self.streams)))
        ]
        for index, stream in enumerate(self.streams):
            bit = 1 << index
            self.length_masks[len(stream)] = self.length_masks.get(len(stream), 0) | bit
            for position, token in enumerate(stream):
                values = token_masks[position]
                values[token] = values.get(token, 0) | bit
        self.token_masks = tuple(token_masks)
        if semantic_records is None:
            records = self.streams if len(self.length_masks) == 1 else None
        else:
            records = tuple(tuple(record) for record in semantic_records)
            if len(records) != self.size:
                raise ValueError("semantic_records must have one row per stream")
            if len({len(record) for record in records}) != 1:
                raise ValueError("semantic_records must have fixed arity")
            try:
                if len(set(records)) != self.size:
                    raise ValueError("semantic_records must be unique")
            except TypeError as error:
                raise ValueError("semantic fields must be hashable") from error
        self.semantic_records = records
        field_masks: list[dict[Hashable, int]] = []
        if records is not None:
            field_masks = [{} for _ in records[0]]
            for index, record in enumerate(records):
                for field, value in enumerate(record):
                    values = field_masks[field]
                    values[value] = values.get(value, 0) | (1 << index)
        self.masks_by_field = tuple(field_masks)

    def _validate_mask(self, mask: int) -> None:
        if type(mask) is not int or mask <= 0 or mask & ~self.full_mask:
            raise ValueError("posterior must be a nonempty subset of the corpus")

    def posterior(
        self,
        tokens: Sequence[Hashable],
        hidden: Sequence[bool],
        *,
        reveal_length: bool = True,
    ) -> int:
        """Intersect observations; hidden positions never inspect their token.

        Known total length is an observation. To withhold it, supply a corpus
        already padded to a common canvas and mask its PAD tokens too.
        An impossible observation returns the empty bitset zero.
        """
        if len(tokens) != len(hidden) or any(type(flag) is not bool for flag in hidden):
            raise ValueError("hidden must contain one boolean per token")
        if not reveal_length and len(self.length_masks) != 1:
            raise ValueError("withheld length requires equal-length padded streams")
        mask = self.length_masks.get(len(tokens), 0)
        for position, (token, is_hidden) in enumerate(zip(tokens, hidden)):
            if mask == 0:
                break
            if not is_hidden:
                try:
                    mask &= self.token_masks[position].get(token, 0)
                except TypeError as error:
                    raise ValueError("observed tokens must be hashable") from error
        return mask

    @lru_cache(maxsize=32768)
    def _posterior_stats(self, mask: int) -> tuple:
        count = mask.bit_count()
        entropy = math.log2(count)
        field_entropies = []
        for values in self.masks_by_field:
            counts = [(mask & value_mask).bit_count() for value_mask in values.values()]
            field_entropies.append(math.fsum(
                -number / count * math.log2(number / count)
                for number in counts if number
            ))
        correlation = None if self.semantic_records is None else max(
            0.0, math.fsum(field_entropies) - entropy,
        )
        return count, entropy, tuple(field_entropies), correlation

    def posterior_stats(self, mask: int) -> dict:
        self._validate_mask(mask)
        count, entropy, fields, correlation = self._posterior_stats(mask)
        return {
            "candidate_count": count,
            "entropy_bits": entropy,
            "bayes_recovery": 1.0 / count,
            "unique_recovery": float(count == 1),
            "field_entropy_bits": list(fields),
            "conditional_total_correlation_bits": correlation,
        }

    def completion_stats(self, mask: int, groups: Sequence[Sequence[int]]) -> dict:
        """Exact law of independent-within-round, conditional-between-round sampling.

        Groups must partition *all* positions of a common-length candidate
        stream. A group's marginals all use the state before that round. The
        returned probabilities sum only valid full completions; omitted mass is
        invalid output. TV includes that invalid mass. Conditioning on valid
        output can still leave a biased posterior, reported separately.
        """
        self._validate_mask(mask)
        lengths = {len(self.streams[index]) for index in _members(mask)}
        if len(lengths) != 1:
            raise ValueError("completion candidates must have the same length")
        length = lengths.pop()
        partition = tuple(tuple(group) for group in groups)
        positions = tuple(position for group in partition for position in group)
        if (not partition or any(not group for group in partition)
                or any(type(position) is not int for position in positions)
                or sorted(positions) != list(range(length))):
            raise ValueError("nonempty groups must partition all token positions exactly once")
        log_probabilities = self._completion_log_probabilities(mask, partition)
        log_mass = min(0.0, _logsumexp(log_probabilities))
        mass = min(1.0, math.exp(log_mass))
        count = mask.bit_count()
        prior = 1.0 / count
        probabilities = [math.exp(probability) for probability in log_probabilities]
        overlap = math.fsum(min(prior, probability) for probability in probabilities)
        accepted_overlap = math.fsum(
            min(prior, math.exp(probability - log_mass))
            for probability in log_probabilities
        )
        retries = geometric_retry_stats(mass)
        # A probability can underflow while its log remains finite. Preserve
        # a representable expected count where possible, otherwise use null.
        expected = retries["expected_retry_rounds"]
        if expected is None and -log_mass <= math.log(sys.float_info.max):
            expected = math.exp(-log_mass)
        rounds = len(partition)
        expected_prediction_rounds = None if expected is None else rounds * expected
        if expected_prediction_rounds is not None and not math.isfinite(expected_prediction_rounds):
            expected_prediction_rounds = None
        attempts_for_confidence = retries["rounds_for_confidence"]
        return {
            "candidate_count": count,
            "number_rounds": rounds,
            "processed_positions": rounds * length,
            "valid_support_mass": mass,
            "log_valid_support_mass": log_mass,
            "total_variation_to_posterior": max(0.0, 1.0 - overlap),
            "accepted_total_variation_to_posterior": max(0.0, 1.0 - accepted_overlap),
            "original_recovery_probability": mass / count,
            "accepted_original_recovery_probability": prior,
            # A retry "round" here is one entire K-round completion attempt.
            # The explicit prediction-round counters include that K multiplier.
            "expected_retry_rounds": expected,
            "rounds_for_99pct_valid": attempts_for_confidence,
            "expected_retry_attempts": expected,
            "attempts_for_99pct_valid": attempts_for_confidence,
            "expected_total_prediction_rounds": expected_prediction_rounds,
            "prediction_rounds_for_99pct_valid": (
                None if attempts_for_confidence is None else rounds * attempts_for_confidence
            ),
        }

    def independent_group_refinement(self, mask: int, order: str = "forward") -> dict:
        """Exact greedy sampling in conditionally independent groups.

        At each round, materialize all coordinates already deterministic under
        the pre-round posterior. In the requested scan order, append uncertain
        coordinates only when each is independent of the *entire joint* of
        coordinates selected so far. The test uses integer contingency counts,
        not pairwise mutual information or a floating-point entropy tolerance.

        Then sample that independent group and branch on its possible values.
        Every branch preserves the exact posterior. Newly implied deterministic
        coordinates need the NEXT prediction round unless they were already in
        the selected group. Round counts are therefore operational prediction
        passes, including final materialization. The greedy order is not claimed
        to minimize passes and assumes an unlimited exact dependency oracle.
        """
        self._validate_mask(mask)
        if order not in ("forward", "reverse"):
            raise ValueError("order must be 'forward' or 'reverse'")
        lengths = {len(self.streams[index]) for index in _members(mask)}
        if len(lengths) != 1:
            raise ValueError("refinement candidates must share a common canvas length")
        length = lengths.pop()
        initial = tuple(range(length))
        if order == "reverse":
            initial = initial[::-1]

        def rounds_by_meaning(candidates: int, remaining: tuple[int, ...]) -> Counter:
            count = candidates.bit_count()
            if not remaining:
                return Counter({0: count})
            if count == 1:
                return Counter({1: 1})
            selected = set()
            uncertain = []
            for position in remaining:
                # Equivalent partitions, including a renaming of token values,
                # carry identical dependence information and can share tests.
                partition = tuple(sorted(
                    intersection for token_mask in self.token_masks[position].values()
                    if (intersection := candidates & token_mask)
                ))
                if len(partition) == 1:
                    selected.add(position)
                else:
                    uncertain.append((position, partition))
            joint = (candidates,)
            tested_partitions = set()
            for position, partition in uncertain:
                if len(joint) == count:
                    # Selected coordinates already identify each candidate;
                    # every remaining uncertain coordinate depends on them.
                    break
                if partition in tested_partitions:
                    continue
                tested_partitions.add(partition)
                refined = []
                independent = True
                for joint_block in joint:
                    joint_count = joint_block.bit_count()
                    for token_block in partition:
                        intersection = joint_block & token_block
                        if (intersection.bit_count() * count
                                != joint_count * token_block.bit_count()):
                            independent = False
                            break
                        refined.append(intersection)
                    if not independent:
                        break
                if independent:
                    selected.add(position)
                    joint = tuple(refined)
            if not selected:
                raise AssertionError("at least the first uncertain position can be selected")
            following_positions = tuple(position for position in remaining if position not in selected)
            if not following_positions:
                return Counter({1: count})
            histogram = Counter()
            for branch in joint:
                for rounds, meanings in rounds_by_meaning(branch, following_positions).items():
                    histogram[rounds + 1] += meanings
            return histogram

        histogram = rounds_by_meaning(mask, initial)
        count = mask.bit_count()
        if sum(histogram.values()) != count:
            raise AssertionError("branch probabilities must exhaust the posterior")
        mean_rounds = math.fsum(rounds * meanings / count for rounds, meanings in histogram.items())
        return {
            "candidate_count": count,
            "order": order,
            "policy": "greedy_exact_conditional_joint_independence_not_optimal",
            "valid_support_mass": 1.0,
            "total_variation_to_posterior": 0.0,
            "original_recovery_probability": 1.0 / count,
            "mean_number_rounds": mean_rounds,
            "worst_number_rounds": max(histogram),
            "probability_by_rounds": [
                {"number_rounds": rounds, "meaning_count": meanings,
                 "probability": meanings / count}
                for rounds, meanings in sorted(histogram.items())
            ],
            "mean_processed_positions": mean_rounds * length,
            "mean_quadratic_work_proxy": mean_rounds * length ** 2,
        }

    @lru_cache(maxsize=4096)
    def _completion_log_probabilities(
        self, mask: int, groups: tuple[tuple[int, ...], ...],
    ) -> tuple[float, ...]:
        # A state is a surviving, valid prefix's candidate set and its proposal
        # log-probability. Invalid prefixes can never regain probability mass.
        states = [(mask, 0.0)]
        for group in groups:
            following = []
            for candidates, log_prefix in states:
                if candidates.bit_count() == 1:
                    following.append((candidates, log_prefix))
                    continue
                count = candidates.bit_count()
                log_count = math.log(count)
                patterns: dict[tuple, int] = {}
                for index in _members(candidates):
                    pattern = tuple(self.streams[index][position] for position in group)
                    patterns[pattern] = patterns.get(pattern, 0) | (1 << index)
                marginal_logs: dict[tuple[int, Hashable], float] = {}
                for pattern, branch in patterns.items():
                    logs = []
                    for position, token in zip(group, pattern):
                        key = (position, token)
                        if key not in marginal_logs:
                            number = (candidates & self.token_masks[position][token]).bit_count()
                            marginal_logs[key] = math.log(number) - log_count
                        logs.append(marginal_logs[key])
                    following.append((branch, log_prefix + math.fsum(logs)))
            states = following
        if any(candidates.bit_count() != 1 for candidates, _ in states):
            raise AssertionError("a complete injective partition must identify each stream")
        return tuple(probability for _, probability in states)

"""Exact finite-world masking and conditional sampling experiments.

This is an information-theoretic oracle, not a diffusion model. A slot is an
atomic synthetic vocabulary item; these encodings are not real text tokenizers.
All probabilities and output distributions are enumerated with Fraction. Only
entropy and JSON presentation use floating point.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from fractions import Fraction
from functools import lru_cache
from itertools import combinations, product
import math
from typing import Mapping

Cell = tuple[int, ...]
Encoded = tuple[Cell, ...]
Observation = tuple[Cell | None, ...]
Distribution = dict[Encoded, Fraction]


@dataclass(frozen=True)
class ToyWorld:
    name: str
    states: tuple[tuple[int, ...], ...]
    description: str


WORLDS = (
    ToyWorld("independent_bits", tuple(product((0, 1), repeat=2)),
             "Two independent fair bits; no dependency benefit is available."),
    ToyWorld("correlated_pair", ((0, 0), (1, 1)),
             "Two fields repeat the same fair bit."),
    ToyWorld("two_correlated_pairs", tuple((a, a, b, b) for a, b in product((0, 1), repeat=2)),
             "Two independent latent bits, each repeated in two fields."),
    ToyWorld("even_parity", ((0, 0, 0), (0, 1, 1), (1, 0, 1), (1, 1, 0)),
             "Three fields have even parity; every pair is marginally independent."),
)


def encoding_groups(field_count: int, name: str) -> tuple[tuple[int, ...], ...]:
    """Map semantic fields to atomic slots, retaining an injective encoding."""
    if name == "factorized":
        return tuple((i,) for i in range(field_count))
    if name == "fused_pairs":
        return tuple(tuple(range(i, min(i + 2, field_count))) for i in range(0, field_count, 2))
    if name == "duplicate_factorized":
        return tuple((i,) for i in range(field_count) for _ in range(2))
    raise ValueError(f"unknown encoding: {name}")


def encoded_prior(world: ToyWorld, encoding: str) -> Distribution:
    groups = encoding_groups(len(world.states[0]), encoding)
    result = {tuple(tuple(state[i] for i in group) for group in groups):
              Fraction(1, len(world.states)) for state in world.states}
    if len(result) != len(world.states):
        raise ValueError("encoding is not injective")
    return result


def entropy(probabilities) -> float:
    return -sum(float(p) * math.log2(float(p)) for p in probabilities if p)


def distribution_metrics(predicted: Mapping[Encoded, Fraction],
                         target: Mapping[Encoded, Fraction]) -> dict[str, float]:
    """Support and TV concern distribution fidelity, not hidden-gold accuracy."""
    keys = set(predicted) | set(target)
    valid = sum((p for value, p in predicted.items() if value in target), Fraction())
    tv = sum((abs(predicted.get(k, Fraction()) - target.get(k, Fraction()))
              for k in keys), Fraction()) / 2
    return {"valid_support_mass": float(valid),
            "total_variation_to_exact_posterior": float(tv)}


class ConditionalOracle:
    """Sample only from the current posterior, never from the hidden source.

    A parallel round samples selected slots independently. We exhaustively find
    the minimum worst-case rounds among policies whose selected slots have an
    exactly factorized joint posterior. Every later round conditions on sampled
    commitments. This policy class guarantees exact final distribution fidelity;
    it does not include remasking, approximate dependencies, or learned errors.
    """

    def __init__(self, prior: Distribution):
        if not prior or sum(prior.values()) != 1 or any(p <= 0 for p in prior.values()):
            raise ValueError("prior must be a normalized positive distribution")
        self.prior = dict(prior)
        self.slot_count = len(next(iter(prior)))
        if any(len(value) != self.slot_count for value in prior):
            raise ValueError("all encodings must have equal slot count")

    @lru_cache(maxsize=None)
    def posterior(self, observation: Observation) -> Distribution:
        if len(observation) != self.slot_count:
            raise ValueError("observation has wrong slot count")
        matches = {value: p for value, p in self.prior.items()
                   if all(seen is None or seen == cell
                          for seen, cell in zip(observation, value))}
        mass = sum(matches.values(), Fraction())
        if not mass:
            raise ValueError("observation has zero probability")
        return {value: p / mass for value, p in matches.items()}

    def parallel_outcomes(self, observation: Observation, slots: tuple[int, ...]):
        posterior = self.posterior(observation)
        marginals = []
        for slot in slots:
            marginal: dict[Cell, Fraction] = defaultdict(Fraction)
            for value, p in posterior.items():
                marginal[value[slot]] += p
            marginals.append(marginal)
        for cells in product(*(tuple(m) for m in marginals)):
            probability = math.prod(marginals[i][cell] for i, cell in enumerate(cells))
            committed = list(observation)
            for slot, cell in zip(slots, cells):
                committed[slot] = cell
            yield tuple(committed), probability

    @lru_cache(maxsize=None)
    def independent(self, observation: Observation, slots: tuple[int, ...]) -> bool:
        joint: dict[tuple[Cell, ...], Fraction] = defaultdict(Fraction)
        for value, p in self.posterior(observation).items():
            joint[tuple(value[i] for i in slots)] += p
        for committed, p in self.parallel_outcomes(observation, slots):
            if joint.get(tuple(committed[i] for i in slots), Fraction()) != p:
                return False
        return True

    @lru_cache(maxsize=None)
    def best_plan(self, observation: Observation) -> tuple[int, tuple[int, ...]]:
        remaining = tuple(i for i, cell in enumerate(observation) if cell is None)
        if not remaining:
            return 0, ()
        if self.independent(observation, remaining):
            return 1, remaining
        best_rounds, best_slots = self.slot_count + 1, ()
        # Larger sets first; a two-round exact plan reaches the proven lower
        # bound when the full remaining set is not independent.
        for count in range(len(remaining) - 1, 0, -1):
            for slots in combinations(remaining, count):
                if not self.independent(observation, slots):
                    continue
                rounds = 1 + max(self.best_plan(committed)[0]
                                 for committed, _ in self.parallel_outcomes(observation, slots))
                if rounds < best_rounds:
                    best_rounds, best_slots = rounds, slots
                if rounds == 2:
                    return best_rounds, best_slots
        return best_rounds, best_slots

    @lru_cache(maxsize=None)
    def sample(self, observation: Observation, round_budget: int) -> tuple[Distribution, Fraction]:
        """Enumerate complete outputs and mean rounds for this bounded policy.

        The final allowed round fills all remaining slots by marginal sampling.
        Earlier rounds use the exact independent-group plan. Thus increasing the
        budget reaches exact fidelity without ever revealing original values.
        """
        remaining = tuple(i for i, cell in enumerate(observation) if cell is None)
        if not remaining:
            return {tuple(observation): Fraction(1)}, Fraction(0)
        if round_budget < 1:
            raise ValueError("a positive round budget is required for masked slots")
        slots = remaining if round_budget == 1 else self.best_plan(observation)[1]
        outputs: Distribution = defaultdict(Fraction)
        rounds = Fraction(1)
        for committed, p in self.parallel_outcomes(observation, slots):
            if round_budget == 1:
                outputs[tuple(committed)] += p
            else:
                downstream, additional_rounds = self.sample(committed, round_budget - 1)
                rounds += p * additional_rounds
                for value, probability in downstream.items():
                    outputs[value] += p * probability
        return dict(outputs), rounds


def observation_distribution(prior: Distribution, mask_probability: Fraction):
    """Enumerate independent Bernoulli slot erasures, coalescing observations."""
    if not 0 <= mask_probability <= 1:
        raise ValueError("mask probability must be in [0, 1]")
    result: dict[Observation, Fraction] = defaultdict(Fraction)
    slot_count = len(next(iter(prior)))
    for hidden in product((False, True), repeat=slot_count):
        count = sum(hidden)
        mask_mass = mask_probability ** count * (1 - mask_probability) ** (slot_count - count)
        if not mask_mass:
            continue
        for value, p in prior.items():
            observed = tuple(None if erase else cell for cell, erase in zip(value, hidden))
            result[observed] += mask_mass * p
    return dict(result)


def evaluate_encoding(world: ToyWorld, encoding: str, mask_probability: Fraction,
                      oracle: ConditionalOracle | None = None) -> dict:
    prior = encoded_prior(world, encoding)
    oracle = oracle or ConditionalOracle(prior)
    observations = observation_distribution(prior, mask_probability)
    worst_rounds = max(oracle.best_plan(observation)[0] for observation in observations)
    budgets = range(1, max(worst_rounds, 1) + 1)
    summary = {budget: {"valid_support_mass": 0.0,
                        "total_variation_to_exact_posterior": 0.0,
                        "expected_executed_rounds": 0.0,
                        "worst_observation_valid_support_mass": 1.0,
                        "worst_observation_total_variation": 0.0} for budget in budgets}
    residual_entropy = recovery = unique = 0.0
    round_histogram: dict[int, Fraction] = defaultdict(Fraction)
    for observation, observation_mass in observations.items():
        posterior = oracle.posterior(observation)
        mass = float(observation_mass)
        residual_entropy += mass * entropy(posterior.values())
        recovery += mass * float(max(posterior.values()))
        unique += mass * (len(posterior) == 1)
        round_histogram[oracle.best_plan(observation)[0]] += observation_mass
        for budget in budgets:
            outputs, executed_rounds = oracle.sample(observation, budget)
            metrics = distribution_metrics(outputs, posterior)
            for name, value in metrics.items():
                summary[budget][name] += mass * value
            summary[budget]["expected_executed_rounds"] += mass * float(executed_rounds)
            summary[budget]["worst_observation_valid_support_mass"] = min(
                summary[budget]["worst_observation_valid_support_mass"], metrics["valid_support_mass"])
            summary[budget]["worst_observation_total_variation"] = max(
                summary[budget]["worst_observation_total_variation"], metrics["total_variation_to_exact_posterior"])
    slot_count = oracle.slot_count
    per_budget = []
    for budget, metrics in summary.items():
        per_budget.append({"round_budget": budget, **metrics,
                           "fixed_canvas_linear_slot_rounds": slot_count * metrics["expected_executed_rounds"],
                           "fixed_canvas_quadratic_slot_rounds": slot_count ** 2 * metrics["expected_executed_rounds"]})
    passing = [row["round_budget"] for row in per_budget
               if row["valid_support_mass"] >= 0.99 - 1e-12
               and row["total_variation_to_exact_posterior"] <= 0.01 + 1e-12]
    worst_passing = [row["round_budget"] for row in per_budget
                     if row["worst_observation_valid_support_mass"] >= 0.99 - 1e-12
                     and row["worst_observation_total_variation"] <= 0.01 + 1e-12]
    return {"world": world.name, "encoding": encoding, "slot_count": slot_count,
            "semantic_field_count": len(world.states[0]), "prior_entropy_bits": entropy(prior.values()),
            "mask_probability": float(mask_probability),
            "expected_masked_slots": slot_count * float(mask_probability),
            "expected_residual_entropy_bits": residual_entropy,
            "bayes_original_recovery_ceiling": recovery,
            "unique_recovery_probability": unique,
            "one_round": {key: summary[1][key] for key in
                          ("valid_support_mass", "total_variation_to_exact_posterior")},
            "optimal_exact_independent_parallel_rounds": {
                "expected": sum(rounds * float(mass) for rounds, mass in round_histogram.items()),
                "worst_case": worst_rounds,
                "probability_by_rounds": {str(rounds): float(mass) for rounds, mass in sorted(round_histogram.items())}},
            "refinement_by_round_budget": per_budget,
            "minimum_tested_round_budget_meeting_threshold": min(passing) if passing else None,
            "minimum_tested_round_budget_meeting_worst_observation_threshold":
                min(worst_passing) if worst_passing else None}


def run_fusion_experiments() -> dict:
    """Return deterministic, JSON-serializable results without filesystem I/O."""
    encodings = ("factorized", "fused_pairs", "duplicate_factorized")
    mask_probabilities = (Fraction(0), Fraction(1, 4), Fraction(1, 2), Fraction(3, 4), Fraction(1))
    rows = []
    for world in WORLDS:
        for encoding in encodings:
            oracle = ConditionalOracle(encoded_prior(world, encoding))
            rows.extend(evaluate_encoding(world, encoding, p, oracle) for p in mask_probabilities)
    return {
        "experiment": "exact_finite_world_fusion_and_refinement",
        "version": 1,
        "methodology": {
            "probabilities": "Exact Fraction enumeration; entropy and exported metrics use float.",
            "prior": "Uniform over the explicitly listed semantic world states.",
            "masking": "Independent Bernoulli erasure of atomic synthetic slots; observed slots are correct.",
            "tokenization": "Synthetic atomic slots, not Sylang/English/JSON text tokenizer counts.",
            "recovery": "Bayes maximum posterior probability of recovering the original hidden world.",
            "fidelity": "TV and valid support compare sampled outputs with the exact conditional distribution.",
            "refinement": "Condition on sampled commitments; no hidden-gold reveal. Parallel slots sample independently.",
            "optimality_scope": "Minimum worst-case rounds within exact posterior-independent group policies; not a bound for all diffusion algorithms.",
            "threshold": {"minimum_valid_support_mass": 0.99, "maximum_total_variation": 0.01},
            "threshold_aggregation": "Primary threshold averages over source worlds and masks; worst-observation threshold is also reported.",
            "budget_rule": "Final allowed round fills remaining slots independently; prior rounds commit an optimal independent subset.",
            "cost_proxy": "Fixed synthetic canvas slots times executed rounds (or squared slots); excludes network/heads/packing/decoder costs.",
            "limits": "Tiny known priors; no learned model, training, speed timing, remasking, or unequal-slot output-head cost.",
        },
        "worlds": [{"name": world.name, "description": world.description,
                    "states": [list(state) for state in world.states]} for world in WORLDS],
        "encodings": {encoding: "Maps listed semantic field indices into each atomic slot."
                      for encoding in encodings},
        "slot_groups_by_field_count": {
            str(count): {encoding: [list(group) for group in encoding_groups(count, encoding)]
                         for encoding in encodings} for count in (2, 3, 4)},
        "mask_probabilities": [float(p) for p in mask_probabilities],
        "observations": rows,
    }

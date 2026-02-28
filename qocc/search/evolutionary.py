"""Evolutionary search strategy for pipeline candidate optimization."""

from __future__ import annotations

from typing import Any

import numpy as np

from qocc import DEFAULT_SEED
from qocc.core.circuit_handle import PipelineSpec
from qocc.search.space import Candidate, SearchSpaceConfig


class EvolutionaryOptimizer:
    """Tournament-selection evolutionary optimizer over pipeline parameter vectors."""

    def __init__(self, config: SearchSpaceConfig) -> None:
        self.config = config
        self.population_size = max(2, int(config.evolutionary_population_size))
        self.mutation_rate = float(config.evolutionary_mutation_rate)
        self.crossover_rate = float(config.evolutionary_crossover_rate)
        self.tournament_size = max(2, int(config.evolutionary_tournament_size))
        self.elitism = max(1, int(config.evolutionary_elitism))
        self.mutation_sigma = float(config.evolutionary_mutation_sigma)

        seed = config.seeds[0] if config.seeds else DEFAULT_SEED
        self.rng = np.random.default_rng(seed)

        self.axes: dict[str, list[Any]] = {
            "optimization_level": list(config.optimization_levels or [1]),
            "seed": list(config.seeds or [DEFAULT_SEED]),
        }
        if config.routing_methods:
            self.axes["routing_method"] = list(config.routing_methods)
        for key, values in (config.extra_params or {}).items():
            if isinstance(values, list) and values:
                self.axes[key] = list(values)

        self.keys = list(self.axes.keys())
        self._seen_ids: set[str] = set()

    def initial_population(self) -> list[Candidate]:
        """Generate initial random population with deduplication."""
        pop: list[Candidate] = []
        attempts = 0
        max_attempts = self.population_size * 10
        while len(pop) < self.population_size and attempts < max_attempts:
            attempts += 1
            candidate = self._random_candidate()
            if candidate.candidate_id in self._seen_ids:
                continue
            self._seen_ids.add(candidate.candidate_id)
            pop.append(candidate)
        return pop

    def next_generation(self, ranked_population: list[Candidate]) -> list[Candidate]:
        """Generate the next population using elitism + tournament/crossover/mutation."""
        if not ranked_population:
            return self.initial_population()

        sorted_pop = sorted(ranked_population, key=lambda c: c.surrogate_score)
        elite_n = min(self.elitism, len(sorted_pop), self.population_size)
        next_pop: list[Candidate] = [sorted_pop[i] for i in range(elite_n)]
        next_ids = {c.candidate_id for c in next_pop}

        attempts = 0
        max_attempts = self.population_size * 50
        while len(next_pop) < self.population_size and attempts < max_attempts:
            attempts += 1
            parent_a = self._tournament_select(sorted_pop)
            parent_b = self._tournament_select(sorted_pop)
            genome_a = self._candidate_to_genome(parent_a)
            genome_b = self._candidate_to_genome(parent_b)
            child_genome = self._crossover(genome_a, genome_b)
            child_genome = self._mutate(child_genome)
            child = self._genome_to_candidate(child_genome)
            if child.candidate_id in next_ids:
                continue
            next_ids.add(child.candidate_id)
            self._seen_ids.add(child.candidate_id)
            next_pop.append(child)

        while len(next_pop) < self.population_size:
            child = self._random_candidate()
            if child.candidate_id in next_ids:
                continue
            next_ids.add(child.candidate_id)
            self._seen_ids.add(child.candidate_id)
            next_pop.append(child)

        return next_pop

    def _random_candidate(self) -> Candidate:
        genome: list[int] = []
        for key in self.keys:
            values = self.axes[key]
            genome.append(int(self.rng.integers(0, len(values))))
        return self._genome_to_candidate(genome)

    def _tournament_select(self, population: list[Candidate]) -> Candidate:
        size = min(self.tournament_size, len(population))
        if size <= 1:
            return population[0]
        idxs = self.rng.choice(len(population), size=size, replace=False)
        contenders = [population[int(i)] for i in idxs]
        return min(contenders, key=lambda c: c.surrogate_score)

    def _candidate_to_genome(self, candidate: Candidate) -> list[int]:
        d = candidate.pipeline.to_dict()
        params = dict(d.get("parameters", {}))
        params["optimization_level"] = candidate.pipeline.optimization_level
        genome: list[int] = []
        for key in self.keys:
            values = self.axes[key]
            value = params.get(key, values[0])
            try:
                genome.append(values.index(value))
            except ValueError:
                genome.append(0)
        return genome

    def _genome_to_candidate(self, genome: list[int]) -> Candidate:
        params: dict[str, Any] = {}
        for i, key in enumerate(self.keys):
            values = self.axes[key]
            idx = max(0, min(int(genome[i]), len(values) - 1))
            params[key] = values[idx]

        opt_level = int(params.pop("optimization_level", 1))
        seed = int(params.pop("seed", DEFAULT_SEED))
        params["seed"] = seed

        pipeline = PipelineSpec(
            adapter=self.config.adapter,
            optimization_level=opt_level,
            parameters=params,
        )
        return Candidate(pipeline=pipeline)

    def _crossover(self, a: list[int], b: list[int]) -> list[int]:
        if len(a) != len(b) or len(a) <= 1 or self.rng.random() >= self.crossover_rate:
            return list(a)
        cut = int(self.rng.integers(1, len(a)))
        return list(a[:cut]) + list(b[cut:])

    def _mutate(self, genome: list[int]) -> list[int]:
        out = list(genome)
        for i, key in enumerate(self.keys):
            if self.rng.random() >= self.mutation_rate:
                continue
            n = len(self.axes[key])
            if n <= 1:
                continue
            delta = int(round(float(self.rng.normal(0.0, self.mutation_sigma * max(1, n - 1)))))
            mutated = max(0, min(n - 1, out[i] + delta))
            if mutated == out[i]:
                choices = [j for j in range(n) if j != out[i]]
                mutated = int(self.rng.choice(choices))
            out[i] = mutated
        return out


def population_diversity(population: list[Candidate]) -> float:
    """Compute simple diversity score in [0, 1] from unique candidate IDs."""
    if not population:
        return 0.0
    unique = len({c.candidate_id for c in population})
    return unique / max(1, len(population))

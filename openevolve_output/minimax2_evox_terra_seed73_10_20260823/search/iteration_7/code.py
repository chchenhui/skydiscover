# EVOLVE-BLOCK-START
import logging
import math
import random
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from skydiscover.config import DatabaseConfig
from skydiscover.search.base_database import Program, ProgramDatabase

logger = logging.getLogger(__name__)


@dataclass
class EvolvedProgram(Program):
    """Program for the evolved database."""


class EvolvedProgramDatabase(ProgramDatabase):
    """Score-aware elite rotation with complementary context selection."""

    def __init__(self, name: str, config: DatabaseConfig):
        super().__init__(name, config)
        self.initial_program = None
        self.random_state = random.Random(getattr(config, "random_seed", None))
        self.best_observed_score = float("-inf")
        self.last_meaningful_improvement_iteration = -1
        self.parent_use_count: Dict[str, int] = {}
        self.context_use_count: Dict[str, int] = {}
        self.score_history: List[float] = []

    def _score(self, program: EvolvedProgram) -> Optional[float]:
        value = program.metrics.get("combined_score")
        if isinstance(value, (int, float)):
            score = float(value)
            if math.isfinite(score):
                return score
        return None

    def add(
        self, program: EvolvedProgram, iteration: Optional[int] = None, **kwargs: Any
    ) -> str:
        if iteration == 0 or program.iteration_found == 0:
            self.initial_program = program

        self.programs[program.id] = program

        if iteration is not None:
            self.last_iteration = max(self.last_iteration, iteration)

        if program.parent_id:
            self.parent_use_count[program.parent_id] = (
                self.parent_use_count.get(program.parent_id, 0) + 1
            )
        for context_id in program.other_context_ids or []:
            self.context_use_count[context_id] = (
                self.context_use_count.get(context_id, 0) + 1
            )

        score = self._score(program)
        if score is not None:
            previous_best = self.best_observed_score
            self.score_history.append(score)
            if score > previous_best:
                meaningful = (
                    previous_best == float("-inf")
                    or score - previous_best > max(0.01, abs(previous_best) * 0.01)
                )
                self.best_observed_score = score
                if meaningful:
                    self.last_meaningful_improvement_iteration = (
                        iteration if iteration is not None else program.iteration_found
                    )

        if self.config.db_path:
            self._save_program(program)
        self._update_best_program(program)

        logger.debug("Added program %s to the evolve database", program.id)
        return program.id

    def sample(
        self, num_context_programs: Optional[int] = 4, **kwargs: Any
    ) -> Tuple[Dict[str, EvolvedProgram], Dict[str, List[EvolvedProgram]]]:
        candidates = list(self.programs.values())
        if not candidates:
            raise ValueError("No candidates available for sampling")

        scored = [(self._score(p), p) for p in candidates]
        valid = [(s, p) for s, p in scored if s is not None]

        if not valid:
            parent = self.random_state.choice(candidates)
            pool = [p for p in candidates if p.id != parent.id]
            self.random_state.shuffle(pool)
            return {"": parent}, {"": pool[: num_context_programs or 0]}

        best_score = max(score for score, _ in valid)
        # Keep the parent at the current frontier, but rotate among equivalent
        # elite solutions rather than repeatedly mutating one representation.
        elite = [
            p for score, p in valid
            if score >= best_score - max(0.001, abs(best_score) * 0.002)
        ]
        least_used = min(self.parent_use_count.get(p.id, 0) for p in elite)
        parent_choices = [
            p for p in elite if self.parent_use_count.get(p.id, 0) == least_used
        ]
        parent = self.random_state.choice(parent_choices)

        limit = num_context_programs or 0
        remaining = [p for _, p in valid if p.id != parent.id]

        # Context deliberately combines alternate frontier solutions with the
        # strongest distinct challengers, rather than four redundant elites.
        remaining.sort(key=lambda p: self._score(p) or float("-inf"), reverse=True)
        alternate_elites = [p for p in remaining if p in elite]
        challengers = [p for p in remaining if p not in elite]
        self.random_state.shuffle(alternate_elites)
        self.random_state.shuffle(challengers)

        contexts: List[EvolvedProgram] = []
        if alternate_elites:
            contexts.append(alternate_elites.pop())
        contexts.extend(challengers[: max(0, limit - len(contexts))])

        if len(contexts) < limit:
            leftovers = alternate_elites + [
                p for p in challengers if p.id not in {x.id for x in contexts}
            ]
            leftovers.sort(key=lambda p: self.context_use_count.get(p.id, 0))
            contexts.extend(leftovers[: limit - len(contexts)])

        return {"": parent}, {"": contexts[:limit]}


# EVOLVE-BLOCK-END
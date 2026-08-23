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
    """Adaptive, diversity-oriented search database."""

    def __init__(self, name: str, config: DatabaseConfig):
        super().__init__(name, config)
        self.initial_program = None
        self.random_state = random.Random(getattr(config, "random_seed", None))
        self.best_seen = float("-inf")
        self.stagnation = 0
        self.parent_uses: Dict[str, int] = {}
        self.parent_successes: Dict[str, int] = {}
        self.parent_gains: Dict[str, float] = {}

    def _score(self, program: EvolvedProgram) -> Optional[float]:
        value = program.metrics.get("combined_score")
        if isinstance(value, (int, float)):
            value = float(value)
            if math.isfinite(value):
                return value
        return None

    def _meaningful(self, new: float, old: float) -> bool:
        return new - old > max(0.01, abs(old) * 0.01)

    def add(
        self, program: EvolvedProgram, iteration: Optional[int] = None, **kwargs: Any
    ) -> str:
        if iteration == 0 or program.iteration_found == 0:
            self.initial_program = program

        self.programs[program.id] = program

        score = self._score(program)
        if score is not None:
            if self.best_seen == float("-inf") or self._meaningful(score, self.best_seen):
                self.best_seen = max(self.best_seen, score)
                self.stagnation = 0
            else:
                self.best_seen = max(self.best_seen, score)
                self.stagnation += 1

        if program.parent_id:
            parent_id = program.parent_id
            self.parent_uses[parent_id] = self.parent_uses.get(parent_id, 0) + 1
            parent = self.get(parent_id)
            parent_score = self._score(parent) if parent is not None else None
            if score is not None and parent_score is not None and self._meaningful(score, parent_score):
                self.parent_successes[parent_id] = self.parent_successes.get(parent_id, 0) + 1
                self.parent_gains[parent_id] = self.parent_gains.get(parent_id, 0.0) + (
                    score - parent_score
                )

        if iteration is not None:
            self.last_iteration = max(self.last_iteration, iteration)
        if self.config.db_path:
            self._save_program(program)
        self._update_best_program(program)
        return program.id

    def sample(
        self, num_context_programs: Optional[int] = 4, **kwargs: Any
    ) -> Tuple[Dict[str, EvolvedProgram], Dict[str, List[EvolvedProgram]]]:
        candidates = list(self.programs.values())
        if not candidates:
            raise ValueError("No candidates available for sampling")

        scored = [(p, self._score(p)) for p in candidates]
        numeric = [(p, s) for p, s in scored if s is not None]
        limit = max(0, num_context_programs or 0)

        # During a plateau, revisit parents which historically produced real gains
        # (the low-scoring seed can be more generative than duplicate elites).
        weights: List[float] = []
        for program, score in scored:
            uses = self.parent_uses.get(program.id, 0)
            successes = self.parent_successes.get(program.id, 0)
            gains = self.parent_gains.get(program.id, 0.0)
            quality = (score or 0.0) if numeric else 0.0

            if self.stagnation >= 3:
                weight = 1.0 + 3.0 * successes + gains - 0.25 * uses
            else:
                weight = 1.0 + max(0.0, quality) - 0.20 * uses + successes
            weights.append(max(0.05, weight))

        parent = self.random_state.choices(candidates, weights=weights, k=1)[0]

        # Context deliberately spans score ranges instead of repeatedly showing
        # nearly identical best programs.
        pool = [(p, s) for p, s in scored if p.id != parent.id]
        self.random_state.shuffle(pool)
        context: List[EvolvedProgram] = []
        seen_solutions = set()

        def add_one(options: List[Tuple[EvolvedProgram, Optional[float]]]) -> None:
            for p, _ in options:
                key = p.solution
                if p.id != parent.id and key not in seen_solutions:
                    context.append(p)
                    seen_solutions.add(key)
                    return

        ranked = sorted(pool, key=lambda item: item[1] if item[1] is not None else float("-inf"))
        if ranked:
            add_one(list(reversed(ranked)))                 # strong reference
            add_one(ranked)                                 # contrasting attempt
            add_one(ranked[len(ranked) // 3:])              # upper-middle idea

        for p, _ in pool:
            if len(context) >= limit:
                break
            if p.solution not in seen_solutions:
                context.append(p)
                seen_solutions.add(p.solution)

        return {"": parent}, {"": context[:limit]}


# EVOLVE-BLOCK-END
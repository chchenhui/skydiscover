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
    """Adaptive score-aware population sampler."""

    def __init__(self, name: str, config: DatabaseConfig):
        super().__init__(name, config)
        self.random_state = random.Random(getattr(config, "random_seed", None))
        self.initial_program = None
        self.best_seen = float("-inf")
        self.stagnant_iterations = 0
        self.last_iteration = getattr(self, "last_iteration", -1)

    def _score(self, program: EvolvedProgram) -> Optional[float]:
        value = program.metrics.get("combined_score")
        if isinstance(value, (int, float)):
            value = float(value)
            if math.isfinite(value):
                return value
        return None

    def add(
        self, program: EvolvedProgram, iteration: Optional[int] = None, **kwargs: Any
    ) -> str:
        if iteration == 0 or program.iteration_found == 0:
            self.initial_program = program

        score = self._score(program)
        if score is not None:
            if self.best_seen == float("-inf"):
                self.best_seen = score
            elif score > self.best_seen:
                improvement = score - self.best_seen
                relative = improvement / max(abs(self.best_seen), 1e-12)
                if improvement > 0.01 or relative > 0.01:
                    self.stagnant_iterations = 0
                else:
                    self.stagnant_iterations += 1
                self.best_seen = score
            else:
                self.stagnant_iterations += 1

        self.programs[program.id] = program
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

        scored = [(self._score(p), p) for p in candidates]
        scored.sort(key=lambda item: item[0] if item[0] is not None else float("-inf"),
                    reverse=True)

        usable = [(s, p) for s, p in scored if s is not None]
        if not usable:
            parent = self.random_state.choice(candidates)
        else:
            best = usable[0][0]
            # Near-best, non-best variants have historically been productive:
            # they preserve strong structure while leaving room for correction.
            near_best = [
                p for s, p in usable
                if s < best and best - s <= max(0.02, abs(best) * 0.03)
            ]
            pool = near_best or [p for _, p in usable[:max(1, min(5, len(usable)))]]

            parent_use = {p.id: 0 for p in pool}
            for child in candidates:
                if child.parent_id in parent_use:
                    parent_use[child.parent_id] += 1

            weights = [1.0 / (1.0 + parent_use[p.id]) for p in pool]
            parent = self.random_state.choices(pool, weights=weights, k=1)[0]

        count = max(0, num_context_programs or 0)
        context: List[EvolvedProgram] = []
        used_ids = {parent.id}

        # Give the model a strong reference plus nearby alternatives, rather
        # than repeatedly showing an arbitrary collection of duplicates.
        for _, candidate in scored:
            if candidate.id not in used_ids:
                context.append(candidate)
                used_ids.add(candidate.id)
            if len(context) >= count:
                break

        return {"": parent}, {"": context}


# EVOLVE-BLOCK-END
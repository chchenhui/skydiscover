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
    """Adaptive near-best search with parent/context reuse control."""

    def __init__(self, name: str, config: DatabaseConfig):
        super().__init__(name, config)
        self.random_state = random.Random(getattr(config, "random_seed", None))
        self.initial_program = None
        self.best_score = None
        self.stagnation = 0
        self.parent_uses: Dict[str, int] = {}
        self.context_uses: Dict[str, int] = {}

    def _score(self, program: EvolvedProgram) -> Optional[float]:
        value = program.metrics.get("combined_score")
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            value = float(value)
            if math.isfinite(value):
                return value
        return None

    def add(
        self, program: EvolvedProgram, iteration: Optional[int] = None, **kwargs: Any
    ) -> str:
        if iteration == 0 or program.iteration_found == 0:
            self.initial_program = program

        self.programs[program.id] = program

        if program.parent_id:
            self.parent_uses[program.parent_id] = self.parent_uses.get(program.parent_id, 0) + 1
        for context_id in program.other_context_ids:
            self.context_uses[context_id] = self.context_uses.get(context_id, 0) + 1

        score = self._score(program)
        if score is not None:
            if self.best_score is None:
                self.best_score = score
            else:
                improvement = score - self.best_score
                meaningful = improvement > 0.01 or (
                    self.best_score != 0 and improvement / abs(self.best_score) > 0.01
                )
                if meaningful:
                    self.best_score = score
                    self.stagnation = 0
                else:
                    self.stagnation += 1

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

        if not numeric:
            parent = self.random_state.choice(candidates)
        else:
            best = max(s for _, s in numeric)

            # Prefer near-best non-elites: they have demonstrated quality but
            # still leave room for a meaningful improvement.
            challengers = [
                p for p, s in numeric
                if s < best and s >= best - max(0.03, abs(best) * 0.03)
            ]
            pool = challengers or [p for p, s in numeric if s >= best - 0.01]
            least_used = min(self.parent_uses.get(p.id, 0) for p in pool)
            pool = [p for p in pool if self.parent_uses.get(p.id, 0) <= least_used + 1]
            parent = self.random_state.choice(pool)

        count = max(0, num_context_programs or 0)
        available = [p for p in candidates if p.id != parent.id]
        contexts: List[EvolvedProgram] = []
        seen_solutions = set()

        # High-score, differently written examples are most useful context for
        # refining a near-best construction.  Lower reuse breaks repeated
        # parent/context combinations during a plateau.
        while available and len(contexts) < count:
            weights = []
            for candidate in available:
                score = self._score(candidate)
                quality = 1.0 if score is None else max(0.05, score)
                reuse = 1.0 + self.context_uses.get(candidate.id, 0)
                duplicate = 0.25 if candidate.solution in seen_solutions else 1.0
                weights.append(quality * duplicate / reuse)

            chosen = self.random_state.choices(available, weights=weights, k=1)[0]
            contexts.append(chosen)
            seen_solutions.add(chosen.solution)
            available = [p for p in available if p.id != chosen.id]

        return {"": parent}, {"": contexts}


# EVOLVE-BLOCK-END
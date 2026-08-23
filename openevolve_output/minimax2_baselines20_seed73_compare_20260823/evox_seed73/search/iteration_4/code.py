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
    """Score-aware search with rotation among strong, underused candidates."""

    def __init__(self, name: str, config: DatabaseConfig):
        super().__init__(name, config)
        self.initial_program = None
        self.random_state = random.Random(getattr(config, "random_seed", None))
        self.best_score = None
        self.no_progress = 0
        self.parent_use: Dict[str, int] = {}
        self.best_history: List[float] = []

    def _score(self, program: EvolvedProgram) -> Optional[float]:
        value = program.metrics.get("combined_score")
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            value = float(value)
            if math.isfinite(value):
                return value
        return None

    def add(self, program: EvolvedProgram, iteration: Optional[int] = None, **kwargs: Any) -> str:
        if iteration == 0 or program.iteration_found == 0:
            self.initial_program = program

        self.programs[program.id] = program

        if program.parent_id:
            self.parent_use[program.parent_id] = self.parent_use.get(program.parent_id, 0) + 1

        score = self._score(program)
        if score is not None:
            if self.best_score is None:
                self.best_score = score
            else:
                meaningful_margin = max(0.01, abs(self.best_score) * 0.01)
                if score > self.best_score + meaningful_margin:
                    self.best_score = score
                    self.no_progress = 0
                else:
                    self.no_progress += 1
                    if score > self.best_score:
                        self.best_score = score
            self.best_history.append(score)

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
        numeric = [(s, p) for s, p in scored if s is not None]

        if not numeric:
            parent = self.random_state.choice(candidates)
            pool = [p for p in candidates if p.id != parent.id]
            self.random_state.shuffle(pool)
            return {"": parent}, {"": pool[:(num_context_programs or 0)]}

        numeric.sort(key=lambda item: item[0], reverse=True)
        best = numeric[0][0]

        # Keep selection near the frontier, but rotate away from repeatedly used
        # parents. This is especially useful when several near-identical scores
        # represent different coordinate constructions.
        near_best = [
            p for score, p in numeric
            if score >= best - max(0.015, abs(best) * 0.015)
        ]
        min_use = min(self.parent_use.get(p.id, 0) for p in near_best)
        choices = [p for p in near_best if self.parent_use.get(p.id, 0) == min_use]
        parent = self.random_state.choice(choices)

        count = max(0, num_context_programs or 0)
        pool = [p for _, p in numeric if p.id != parent.id]
        contexts: List[EvolvedProgram] = []

        # Supply strong alternatives first, avoiding duplicate solution text.
        seen_solutions = {parent.solution}
        for p in pool:
            if p.solution not in seen_solutions:
                contexts.append(p)
                seen_solutions.add(p.solution)
            if len(contexts) >= count:
                break

        # If solutions are duplicated, still fill requested context slots.
        if len(contexts) < count:
            for p in pool:
                if p.id not in {q.id for q in contexts}:
                    contexts.append(p)
                if len(contexts) >= count:
                    break

        return {"": parent}, {"": contexts[:count]}


# EVOLVE-BLOCK-END
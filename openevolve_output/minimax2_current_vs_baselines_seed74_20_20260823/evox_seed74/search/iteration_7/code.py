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
    """Adaptive elite search with occasional targeted plateau escapes."""

    def __init__(self, name: str, config: DatabaseConfig):
        super().__init__(name, config)
        self.random_state = random.Random(getattr(config, "random_seed", None))
        self.best_score_seen: Optional[float] = None
        self.last_meaningful_improvement = 0
        self.parent_use: Dict[str, int] = {}
        self.label_use: Dict[str, int] = {
            self.DIVERGE_LABEL: 0,
            self.REFINE_LABEL: 0,
        }

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
        self.programs[program.id] = program

        current_iteration = iteration
        if current_iteration is None:
            current_iteration = program.iteration_found
        if isinstance(current_iteration, int):
            self.last_iteration = max(self.last_iteration, current_iteration)

        if program.parent_id:
            self.parent_use[program.parent_id] = self.parent_use.get(program.parent_id, 0) + 1

        label = program.parent_info[0] if program.parent_info else ""
        if label in self.label_use:
            self.label_use[label] += 1

        score = self._score(program)
        if score is not None:
            if self.best_score_seen is None:
                self.best_score_seen = score
                self.last_meaningful_improvement = current_iteration or 0
            else:
                improvement = score - self.best_score_seen
                meaningful = (
                    improvement > 0.01
                    or improvement > 0.01 * abs(self.best_score_seen)
                )
                if meaningful:
                    self.last_meaningful_improvement = current_iteration or 0
                if score > self.best_score_seen:
                    self.best_score_seen = score

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
            return {"": parent}, {"": []}

        best = max(s for s, _ in numeric)
        elite = [
            p for s, p in numeric
            if s >= best - max(0.005, abs(best) * 0.02)
        ]
        if not elite:
            elite = [max(numeric, key=lambda item: item[0])[1]]

        current_iteration = max(
            [self.last_iteration] +
            [p.iteration_found for p in candidates if isinstance(p.iteration_found, int)]
        )
        stalled = current_iteration - self.last_meaningful_improvement >= 4

        def least_used(pool: List[EvolvedProgram]) -> EvolvedProgram:
            minimum = min(self.parent_use.get(p.id, 0) for p in pool)
            choices = [p for p in pool if self.parent_use.get(p.id, 0) == minimum]
            return self.random_state.choice(choices)

        # On a real plateau, alternate between precise elite refinement and
        # a near-elite alternate approach rather than repeatedly sampling one
        # already saturated best solution.
        if stalled:
            if self.label_use[self.REFINE_LABEL] <= self.label_use[self.DIVERGE_LABEL]:
                parent = least_used(elite)
                return {self.REFINE_LABEL: parent}, {}

            alternatives = [p for s, p in numeric if best - 0.08 <= s < best - 0.001]
            parent = least_used(alternatives or elite)
            return {self.DIVERGE_LABEL: parent}, {}

        parent = least_used(elite)

        count = max(0, num_context_programs or 0)
        remaining = [p for p in candidates if p.id != parent.id]
        self.random_state.shuffle(remaining)

        # Preserve useful high-quality examples, but deliberately include one
        # contrasting non-elite solution when available.
        high = [p for p in remaining if p in elite]
        other = [p for p in remaining if p not in elite]
        self.random_state.shuffle(high)
        self.random_state.shuffle(other)

        context: List[EvolvedProgram] = []
        if other and count:
            context.append(other[0])
        for p in high + other[1:]:
            if len(context) >= count:
                break
            if p.id not in [q.id for q in context]:
                context.append(p)

        return {"": parent}, {"": context}


# EVOLVE-BLOCK-END
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
    """Adaptive elite-rotation search for scalar optimization."""

    def __init__(self, name: str, config: DatabaseConfig):
        super().__init__(name, config)
        self.random_state = random.Random(getattr(config, "random_seed", None))
        self.best_seen = float("-inf")
        self.last_meaningful_improvement = 0
        self.add_count = 0

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
        score = self._score(program)
        self.add_count += 1

        if score is not None:
            improvement = score - self.best_seen
            relative = improvement / max(abs(self.best_seen), 1e-12)
            if self.best_seen == float("-inf") or (
                improvement > 0.01 and relative > 0.01
            ):
                self.last_meaningful_improvement = self.add_count
            self.best_seen = max(self.best_seen, score)

        self.programs[program.id] = program
        if iteration is not None:
            self.last_iteration = max(getattr(self, "last_iteration", 0), iteration)

        if self.config.db_path:
            self._save_program(program)
        self._update_best_program(program)
        return program.id

    def sample(
        self, num_context_programs: Optional[int] = 4, **kwargs: Any
    ) -> Tuple[Dict[str, EvolvedProgram], Dict[str, List[EvolvedProgram]]]:
        candidates = list(self.programs.values())
        scored = [(p, self._score(p)) for p in candidates]
        scored = [(p, s) for p, s in scored if s is not None]
        if not scored:
            raise ValueError("No scored candidates available for sampling")

        best = max(s for _, s in scored)

        # Rotate among exact best programs, preferring those that have been
        # mutated least often rather than repeatedly selecting one champion.
        elite = [p for p, s in scored if s >= best - 1e-12]

        def parent_uses(p: EvolvedProgram) -> int:
            return sum(1 for child in candidates if child.parent_id == p.id)

        minimum_uses = min(parent_uses(p) for p in elite)
        parent = self.random_state.choice(
            [p for p in elite if parent_uses(p) == minimum_uses]
        )

        count = max(0, num_context_programs or 0)
        remaining = [(p, s) for p, s in scored if p.id != parent.id]
        top = [p for p, s in remaining if s >= best - 1e-12]
        near = [p for p, s in remaining if s < best - 1e-12 and s >= best * 0.95]

        self.random_state.shuffle(top)
        self.random_state.shuffle(near)

        # Show both successful variants and nearly-successful alternatives:
        # the latter are useful structural hints for escaping the 1.0 plateau.
        context = top[: max(0, count // 2)]
        context.extend(near[: max(0, count - len(context))])

        if len(context) < count:
            used = [p.id for p in context]
            extras = [p for p, _ in remaining if p.id not in used]
            self.random_state.shuffle(extras)
            context.extend(extras[: count - len(context)])

        return {"": parent}, {"": context[:count]}


# EVOLVE-BLOCK-END
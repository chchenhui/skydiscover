# EVOLVE-BLOCK-START
import logging
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
    """Adaptive near-best exploration with mixed-quality context."""

    def __init__(self, name: str, config: DatabaseConfig):
        super().__init__(name, config)
        self.random_state = random.Random(getattr(config, "random_seed", None))
        self.best_seen: Optional[float] = None
        self.stagnant_steps = 0
        self.parent_usage: Dict[str, int] = {}

    def _score(self, program: EvolvedProgram) -> Optional[float]:
        value = program.metrics.get("combined_score") if isinstance(program.metrics, dict) else None
        if isinstance(value, (int, float)):
            value = float(value)
            if value == value:  # exclude NaN
                return value
        return None

    def add(self, program: EvolvedProgram, iteration: Optional[int] = None, **kwargs: Any) -> str:
        self.programs[program.id] = program

        if program.parent_id:
            self.parent_usage[program.parent_id] = self.parent_usage.get(program.parent_id, 0) + 1

        score = self._score(program)
        if score is not None:
            if self.best_seen is None:
                self.best_seen = score
            else:
                meaningful = score > self.best_seen + max(0.01, abs(self.best_seen) * 0.01)
                if meaningful:
                    self.best_seen = score
                    self.stagnant_steps = 0
                else:
                    self.best_seen = max(self.best_seen, score)
                    self.stagnant_steps += 1

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
        if not candidates:
            raise ValueError("No candidates available for sampling")

        scored = [(p, self._score(p)) for p in candidates]
        valid = [(p, s) for p, s in scored if s is not None]
        if not valid:
            parent = self.random_state.choice(candidates)
            return {"": parent}, {"": []}

        best = max(s for _, s in valid)

        # The observed productive tier is just below the incumbent: it retains
        # a useful alternative construction while avoiding repeated best ties.
        near = [p for p, s in valid if s < best and s >= best * 0.97]
        parent_pool = near or [p for p, s in valid if s == best]

        uses = {
            p.id: max(self.parent_usage.get(p.id, 0),
                      sum(1 for child in candidates if child.parent_id == p.id))
            for p in parent_pool
        }
        least_used = min(uses.values())
        parent = self.random_state.choice(
            [p for p in parent_pool if uses[p.id] == least_used]
        )

        count = max(0, num_context_programs or 0)
        high = [p for p, s in valid if p.id != parent.id and s == best]
        alternatives = [p for p in near if p.id != parent.id]
        self.random_state.shuffle(high)
        self.random_state.shuffle(alternatives)

        # Deliberately mix incumbent examples with near-best alternatives.
        context: List[EvolvedProgram] = []
        while len(context) < count and (high or alternatives):
            source = high if len(context) % 2 == 0 and high else alternatives
            if not source:
                source = high
            context.append(source.pop())

        remaining = [p for p in candidates if p.id != parent.id and p.id not in {x.id for x in context}]
        self.random_state.shuffle(remaining)
        context.extend(remaining[:count - len(context)])

        return {"": parent}, {"": context[:count]}


# EVOLVE-BLOCK-END
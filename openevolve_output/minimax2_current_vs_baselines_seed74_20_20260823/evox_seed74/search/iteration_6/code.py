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
    """Adaptive elite search with varied parent and context selection."""

    def __init__(self, name: str, config: DatabaseConfig):
        super().__init__(name, config)
        self.random_state = random.Random(getattr(config, "random_seed", None))
        self.best_seen = None
        self.stagnation = 0
        self.parent_uses: Dict[str, int] = {}

    def add(
        self, program: EvolvedProgram, iteration: Optional[int] = None, **kwargs: Any
    ) -> str:
        self.programs[program.id] = program

        if iteration is not None:
            self.last_iteration = max(getattr(self, "last_iteration", 0), iteration)

        if program.parent_id:
            self.parent_uses[program.parent_id] = (
                self.parent_uses.get(program.parent_id, 0) + 1
            )

        value = program.metrics.get("combined_score")
        if isinstance(value, (int, float)) and math.isfinite(float(value)):
            score = float(value)
            if self.best_seen is None:
                self.best_seen = score
            elif score > self.best_seen:
                gain = score - self.best_seen
                meaningful = gain >= 0.01 or gain >= abs(self.best_seen) * 0.01
                self.best_seen = score
                self.stagnation = 0 if meaningful else self.stagnation + 1
            else:
                self.stagnation += 1

        if self.config.db_path:
            self._save_program(program)
        self._update_best_program(program)
        return program.id

    def sample(
        self, num_context_programs: Optional[int] = 4, **kwargs: Any
    ) -> Tuple[Dict[str, EvolvedProgram], Dict[str, List[EvolvedProgram]]]:
        scored = []
        for program in self.programs.values():
            value = program.metrics.get("combined_score")
            if isinstance(value, (int, float)) and math.isfinite(float(value)):
                scored.append((float(value), program))

        if not scored:
            raise ValueError("No numeric candidates available for sampling")

        scored.sort(key=lambda item: item[0], reverse=True)
        best = scored[0][0]

        # Concentrate on near-best programs, but rotate among them rather than
        # repeatedly mutating one identical elite.
        elite = [p for s, p in scored if s >= best * 0.98]
        if not elite:
            elite = [p for _, p in scored[:1]]

        least_used = min(self.parent_uses.get(p.id, 0) for p in elite)
        parent_pool = [
            p for p in elite if self.parent_uses.get(p.id, 0) == least_used
        ]
        parent = self.random_state.choice(parent_pool)

        count = max(0, num_context_programs or 0)
        contexts: List[EvolvedProgram] = []
        seen_solutions = {parent.solution}

        # Give the model distinct strong constructions first.  A slightly worse
        # near-elite can expose a useful alternative geometric arrangement.
        pool = [p for _, p in scored if p.id != parent.id]
        self.random_state.shuffle(pool)
        pool.sort(
            key=lambda p: next(s for s, q in scored if q.id == p.id),
            reverse=True,
        )

        for program in pool:
            if len(contexts) >= count:
                break
            if program.solution not in seen_solutions:
                contexts.append(program)
                seen_solutions.add(program.solution)

        if len(contexts) < count:
            for program in pool:
                if len(contexts) >= count:
                    break
                if program.id not in [p.id for p in contexts]:
                    contexts.append(program)

        return {"": parent}, {"": contexts}


# EVOLVE-BLOCK-END
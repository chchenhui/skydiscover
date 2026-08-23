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
    """Adaptive database emphasizing underused strong candidates."""

    def __init__(self, name: str, config: DatabaseConfig):
        super().__init__(name, config)
        self.initial_program = None
        self.random_state = random.Random(getattr(config, "random_seed", None))
        self.best_score: Optional[float] = None
        self.stagnation = 0
        self.parent_uses: Dict[str, int] = {}
        self.label_targets: Dict[str, int] = {}

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

        self.programs[program.id] = program

        if program.parent_id:
            self.parent_uses[program.parent_id] = (
                self.parent_uses.get(program.parent_id, 0) + 1
            )

        if isinstance(program.parent_info, tuple) and len(program.parent_info) == 2:
            label, parent_id = program.parent_info
            if label in (self.DIVERGE_LABEL, self.REFINE_LABEL) and parent_id:
                self.label_targets[parent_id] = self.label_targets.get(parent_id, 0) + 1

        score = self._score(program)
        if score is not None:
            if self.best_score is None:
                self.best_score = score
            else:
                improvement = score - self.best_score
                meaningful = (
                    improvement > 0.01
                    or improvement > 0.01 * abs(self.best_score)
                )
                if meaningful:
                    self.best_score = score
                    self.stagnation = 0
                else:
                    self.stagnation += 1
                    if score > self.best_score:
                        self.best_score = score
        else:
            self.stagnation += 1

        if iteration is not None:
            self.last_iteration = max(self.last_iteration, iteration)

        if self.config.db_path:
            self._save_program(program)

        self._update_best_program(program)
        return program.id

    def sample(
        self, num_context_programs: Optional[int] = 4, **kwargs
    ) -> Tuple[Dict[str, EvolvedProgram], Dict[str, List[EvolvedProgram]]]:
        candidates = list(self.programs.values())
        if not candidates:
            raise ValueError("No candidates available for sampling")

        scored = [(self._score(p), p) for p in candidates]
        numeric = [(score, p) for score, p in scored if score is not None]

        if numeric:
            best = max(score for score, _ in numeric)
            # Treat near-best solutions as equally promising, then avoid
            # repeatedly mutating the same member of that tier.
            strong = [p for score, p in numeric if score >= best - 0.01]
        else:
            strong = candidates[:]

        least_used = min(self.parent_uses.get(p.id, 0) for p in strong)
        choices = [
            p for p in strong if self.parent_uses.get(p.id, 0) == least_used
        ]
        parent = self.random_state.choice(choices)

        # Several failed attempts from equally scoring solutions indicate a
        # plateau. Ask for a clean new geometric direction rather than adding
        # distracting examples from the same plateau.
        if self.stagnation >= 2:
            unlabelled = [
                p for p in choices if self.label_targets.get(p.id, 0) == 0
            ]
            if unlabelled:
                parent = self.random_state.choice(unlabelled)
            return {self.DIVERGE_LABEL: parent}, {}

        count = max(0, num_context_programs or 0)
        context_pool = [p for p in strong if p.id != parent.id]
        self.random_state.shuffle(context_pool)
        return {"": parent}, {"": context_pool[:count]}


# EVOLVE-BLOCK-END
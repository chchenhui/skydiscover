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
    """Adaptive elite-refinement search database."""

    def __init__(self, name: str, config: DatabaseConfig):
        super().__init__(name, config)
        self.initial_program = None
        self.random_state = random.Random(getattr(config, "random_seed", None))

        # State is updated when children are added, rather than when sampled.
        self.best_score: Optional[float] = None
        self.last_meaningful_iteration = -1
        self.latest_iteration = 0
        self.parent_use: Dict[str, int] = {}
        self.refine_use: Dict[str, int] = {}

    def _score(self, program: EvolvedProgram) -> Optional[float]:
        value = program.metrics.get("combined_score")
        if not isinstance(value, (int, float)):
            return None
        value = float(value)
        return value if math.isfinite(value) else None

    def add(
        self, program: EvolvedProgram, iteration: Optional[int] = None, **kwargs: Any
    ) -> str:
        """Store programs and learn which parents/labels have already been used."""
        is_new = program.id not in self.programs

        if iteration == 0 or program.iteration_found == 0:
            self.initial_program = program

        self.programs[program.id] = program

        event_iteration = iteration
        if event_iteration is None:
            event_iteration = program.iteration_found
        if isinstance(event_iteration, int):
            self.latest_iteration = max(self.latest_iteration, event_iteration)
            self.last_iteration = max(self.last_iteration, event_iteration)

        if is_new:
            if program.parent_id:
                self.parent_use[program.parent_id] = (
                    self.parent_use.get(program.parent_id, 0) + 1
                )

            parent_info = program.parent_info
            if (
                isinstance(parent_info, tuple)
                and len(parent_info) >= 2
                and parent_info[0] == self.REFINE_LABEL
                and isinstance(parent_info[1], str)
            ):
                parent_id = parent_info[1]
                self.refine_use[parent_id] = self.refine_use.get(parent_id, 0) + 1

            score = self._score(program)
            if score is not None:
                if self.best_score is None:
                    self.best_score = score
                    self.last_meaningful_iteration = self.latest_iteration
                else:
                    improvement = score - self.best_score
                    # Significant means >1% relative OR >0.01 absolute.
                    significant = (
                        improvement > 0.01
                        or improvement > 0.01 * max(abs(self.best_score), 1e-12)
                    )
                    if significant:
                        self.best_score = score
                        self.last_meaningful_iteration = self.latest_iteration
                    elif score > self.best_score:
                        self.best_score = score

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

        scored = [(self._score(program), program) for program in candidates]
        valid = [(score, program) for score, program in scored if score is not None]

        if not valid:
            parent = self.random_state.choice(candidates)
            return {"": parent}, {"": []}

        best = max(score for score, _ in valid)
        # Include ties and near-ties: repeated 1.0 candidates may encode
        # different constructions even when their scores are identical.
        elite = [
            program for score, program in valid
            if score >= best - max(0.005, abs(best) * 0.005)
        ]

        stagnation = self.latest_iteration - self.last_meaningful_iteration
        if stagnation >= 3 and elite:
            # A stalled, high-quality population is best handled by a focused
            # refinement of an elite not already repeatedly refined.
            least_refined = min(self.refine_use.get(p.id, 0) for p in elite)
            options = [
                p for p in elite if self.refine_use.get(p.id, 0) == least_refined
            ]
            least_used = min(self.parent_use.get(p.id, 0) for p in options)
            options = [p for p in options if self.parent_use.get(p.id, 0) == least_used]
            parent = self.random_state.choice(options)
            return {self.REFINE_LABEL: parent}, {"": []}

        # Before stagnation, exploit good candidates while rotating among them.
        least_used = min(self.parent_use.get(p.id, 0) for p in elite)
        options = [p for p in elite if self.parent_use.get(p.id, 0) == least_used]
        parent = self.random_state.choice(options)

        try:
            context_limit = max(0, int(num_context_programs or 0))
        except (TypeError, ValueError):
            context_limit = 0

        pool = [p for _, p in valid if p.id != parent.id]
        self.random_state.shuffle(pool)
        pool.sort(key=lambda p: self._score(p) or float("-inf"), reverse=True)

        # Retain one contrasting non-elite example when available; it can
        # provide a different construction rather than four duplicate elites.
        context = pool[:context_limit]
        non_elite = [p for p in pool if p not in elite]
        if context_limit and non_elite and all(p in elite for p in context):
            context[-1] = self.random_state.choice(non_elite)

        return {"": parent}, {"": context}


# EVOLVE-BLOCK-END
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
    """Adaptive elite search with plateau-directed refinement."""

    def __init__(self, name: str, config: DatabaseConfig):
        super().__init__(name, config)
        self.initial_program = None
        self.random_state = random.Random(getattr(config, "random_seed", None))
        self.best_score: Optional[float] = None
        self.last_improvement_iteration = -1
        self.current_iteration = -1
        self.parent_uses: Dict[str, int] = {}
        self.label_uses: Dict[str, int] = {}
        self.added_ids: Dict[str, bool] = {}

    def _score(self, program: EvolvedProgram) -> Optional[float]:
        value = program.metrics.get("combined_score")
        if not isinstance(value, (int, float)):
            return None
        value = float(value)
        return value if math.isfinite(value) else None

    def add(
        self, program: EvolvedProgram, iteration: Optional[int] = None, **kwargs: Any
    ) -> str:
        if iteration == 0 or program.iteration_found == 0:
            self.initial_program = program

        is_new = program.id not in self.added_ids
        self.programs[program.id] = program
        self.added_ids[program.id] = True

        observed_iteration = iteration
        if observed_iteration is None:
            observed_iteration = program.iteration_found
        if isinstance(observed_iteration, int):
            self.current_iteration = max(self.current_iteration, observed_iteration)
            self.last_iteration = max(self.last_iteration, observed_iteration)

        if is_new:
            if program.parent_id:
                self.parent_uses[program.parent_id] = (
                    self.parent_uses.get(program.parent_id, 0) + 1
                )

            parent_label = program.parent_info[0] if program.parent_info else ""
            if parent_label:
                self.label_uses[parent_label] = self.label_uses.get(parent_label, 0) + 1

            score = self._score(program)
            if score is not None:
                if self.best_score is None:
                    self.best_score = score
                    self.last_improvement_iteration = self.current_iteration
                elif score > self.best_score:
                    gain = score - self.best_score
                    relative_gain = gain / max(abs(self.best_score), 1e-12)
                    if gain > 0.01 or relative_gain > 0.01:
                        self.last_improvement_iteration = self.current_iteration
                    self.best_score = score

        if self.config.db_path:
            self._save_program(program)

        self._update_best_program(program)
        logger.debug("Added program %s", program.id)
        return program.id

    def sample(
        self, num_context_programs: Optional[int] = 4, **kwargs: Any
    ) -> Tuple[Dict[str, EvolvedProgram], Dict[str, List[EvolvedProgram]]]:
        scored = [(self._score(p), p) for p in self.programs.values()]
        scored = [(score, p) for score, p in scored if score is not None]

        if not scored:
            candidates = list(self.programs.values())
            if not candidates:
                raise ValueError("No candidates available for sampling")
            parent = self.random_state.choice(candidates)
            return {"": parent}, {"": []}

        scored.sort(key=lambda item: item[0], reverse=True)
        elite_index = min(max(1, len(scored) // 2) - 1, len(scored) - 1)
        elite_cutoff = scored[elite_index][0]
        elites = [p for score, p in scored if score >= elite_cutoff]

        # Prefer strong programs, but cycle among underused elite parents.
        least_used = min(self.parent_uses.get(p.id, 0) for p in elites)
        parent_choices = [
            p for p in elites if self.parent_uses.get(p.id, 0) == least_used
        ]
        parent = self.random_state.choice(parent_choices)

        stalled = (
            self.last_improvement_iteration >= 0
            and self.current_iteration - self.last_improvement_iteration >= 2
        )

        # A plateau among several equally strong solutions is a signal to
        # explicitly polish one approach, then try a new direction if needed.
        if stalled:
            refine_count = self.label_uses.get(self.REFINE_LABEL, 0)
            diverge_count = self.label_uses.get(self.DIVERGE_LABEL, 0)
            label = self.REFINE_LABEL if refine_count <= diverge_count else self.DIVERGE_LABEL
            return {label: parent}, {}

        context_count = max(0, num_context_programs or 0)
        pool = [p for _, p in scored if p.id != parent.id]
        self.random_state.shuffle(pool)

        # Keep at least one high-quality alternative near the front.
        elite_context = [p for p in elites if p.id != parent.id]
        self.random_state.shuffle(elite_context)
        context: List[EvolvedProgram] = elite_context[:context_count]

        used_ids = {p.id for p in context}
        for candidate in pool:
            if len(context) >= context_count:
                break
            if candidate.id not in used_ids:
                context.append(candidate)
                used_ids.add(candidate.id)

        return {"": parent}, {"": context}


# EVOLVE-BLOCK-END
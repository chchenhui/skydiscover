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
    """Adaptive elite/diversity search for small program populations."""

    def __init__(self, name: str, config: DatabaseConfig):
        super().__init__(name, config)
        self.random_state = random.Random(getattr(config, "random_seed", None))
        self.best_score_seen: Optional[float] = None
        self.stagnant_additions = 0
        self.parent_use_count: Dict[str, int] = {}
        self.label_use_count: Dict[str, int] = {
            self.DIVERGE_LABEL: 0,
            self.REFINE_LABEL: 0,
        }

    def _score(self, program: EvolvedProgram) -> Optional[float]:
        value = program.metrics.get("combined_score")
        if isinstance(value, (int, float)):
            score = float(value)
            if math.isfinite(score):
                return score
        return None

    def add(
        self,
        program: EvolvedProgram,
        iteration: Optional[int] = None,
        **kwargs: Any,
    ) -> str:
        is_new = program.id not in self.programs
        self.programs[program.id] = program

        if iteration is not None:
            self.last_iteration = max(getattr(self, "last_iteration", 0), iteration)

        if is_new:
            if program.parent_id:
                self.parent_use_count[program.parent_id] = (
                    self.parent_use_count.get(program.parent_id, 0) + 1
                )

            if program.parent_info and len(program.parent_info) >= 1:
                label = program.parent_info[0]
                if label in self.label_use_count:
                    self.label_use_count[label] += 1

            score = self._score(program)
            if score is not None:
                if self.best_score_seen is None:
                    self.best_score_seen = score
                    self.stagnant_additions = 0
                else:
                    threshold = max(0.01, abs(self.best_score_seen) * 0.01)
                    if score > self.best_score_seen + threshold:
                        self.best_score_seen = score
                        self.stagnant_additions = 0
                    else:
                        if score > self.best_score_seen:
                            self.best_score_seen = score
                        self.stagnant_additions += 1

        if self.config.db_path:
            self._save_program(program)
        self._update_best_program(program)

        logger.debug("Added program %s to the evolve database", program.id)
        return program.id

    def sample(
        self,
        num_context_programs: Optional[int] = 4,
        **kwargs: Any,
    ) -> Tuple[Dict[str, EvolvedProgram], Dict[str, List[EvolvedProgram]]]:
        candidates = list(self.programs.values())
        if not candidates:
            raise ValueError("No candidates available for sampling")

        scored = [(program, self._score(program)) for program in candidates]
        numeric = [(program, score) for program, score in scored if score is not None]

        if numeric:
            best = max(score for _, score in numeric)
            # Near-best programs are exploitation candidates.  With the current
            # tied population this deliberately rotates among equally good roots.
            elite = [
                program
                for program, score in numeric
                if score >= best - max(0.01, abs(best) * 0.01)
            ]
        else:
            elite = candidates[:]

        minimum_use = min(self.parent_use_count.get(p.id, 0) for p in elite)
        parent_pool = [
            p for p in elite if self.parent_use_count.get(p.id, 0) == minimum_use
        ]
        parent = self.random_state.choice(parent_pool)

        # Several unchanged elite results indicate a plateau.  Ask for a new
        # direction, balancing divergence/refinement labels across the search.
        if self.stagnant_additions >= 3:
            if self.label_use_count[self.DIVERGE_LABEL] <= self.label_use_count[
                self.REFINE_LABEL
            ]:
                label = self.DIVERGE_LABEL
            else:
                label = self.REFINE_LABEL
            return {label: parent}, {}

        requested = 0 if num_context_programs is None else max(0, num_context_programs)
        available = [p for p in candidates if p.id != parent.id]

        # Context should show distinct score bands and distinct solutions rather
        # than repeatedly showing equivalent top-scoring programs.
        self.random_state.shuffle(available)
        available.sort(
            key=lambda p: (
                self._score(p) is None,
                -(self._score(p) if self._score(p) is not None else float("-inf")),
            )
        )

        context: List[EvolvedProgram] = []
        seen_solutions = {parent.solution}
        for program in available:
            if program.solution not in seen_solutions:
                context.append(program)
                seen_solutions.add(program.solution)
            if len(context) >= requested:
                break

        if len(context) < requested:
            used_ids = {p.id for p in context}
            for program in available:
                if program.id not in used_ids:
                    context.append(program)
                    used_ids.add(program.id)
                if len(context) >= requested:
                    break

        return {"": parent}, {"": context[:requested]}


# EVOLVE-BLOCK-END
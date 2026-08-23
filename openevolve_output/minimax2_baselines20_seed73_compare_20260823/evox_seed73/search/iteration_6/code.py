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
    """Adaptive elite search with occasional plateau-breaking mutations."""

    def __init__(self, name: str, config: DatabaseConfig):
        super().__init__(name, config)
        self.random_state = random.Random(getattr(config, "random_seed", None))
        self.best_seen = float("-inf")
        self.stagnation = 0
        self.initial_program = None

    def _score(self, program: EvolvedProgram) -> Optional[float]:
        value = program.metrics.get("combined_score")
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return float(value)
        return None

    def add(
        self, program: EvolvedProgram, iteration: Optional[int] = None, **kwargs: Any
    ) -> str:
        if iteration == 0 or program.iteration_found == 0:
            self.initial_program = program

        score = self._score(program)
        if score is not None:
            threshold = max(0.01, abs(self.best_seen) * 0.01) if self.best_seen != float("-inf") else 0.01
            if self.best_seen == float("-inf") or score > self.best_seen + threshold:
                self.best_seen = score
                self.stagnation = 0
            else:
                self.stagnation += 1

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
        if not candidates:
            raise ValueError("No candidates available for sampling")

        count = max(0, num_context_programs or 0)
        scored = [(self._score(p), p) for p in candidates]
        valid = [(s, p) for s, p in scored if s is not None]
        valid.sort(key=lambda item: item[0], reverse=True)

        # Prefer elite parents, rotating among equally strong variants that have
        # produced fewer children.
        if valid:
            best_score = valid[0][0]
            elite = [p for s, p in valid if s >= best_score - 0.002]
        else:
            elite = candidates[:]

        child_counts: Dict[str, int] = {}
        for p in candidates:
            if p.parent_id:
                child_counts[p.parent_id] = child_counts.get(p.parent_id, 0) + 1

        least_used = min(child_counts.get(p.id, 0) for p in elite)
        parent_choices = [p for p in elite if child_counts.get(p.id, 0) == least_used]
        parent = self.random_state.choice(parent_choices)

        label = ""
        # A long plateau warrants one targeted change of direction, but labels
        # are deliberately sparse rather than applied on every attempt.
        if self.stagnation >= 8:
            phase = self.stagnation % 6
            if phase in (0, 1):
                label = self.DIVERGE_LABEL
            elif phase == 3:
                label = self.REFINE_LABEL

        if label:
            return {label: parent}, {}

        # Context emphasizes near-elite alternatives: these preserve useful
        # geometry while exposing the LLM to variants that differ from parent.
        pool = [p for _, p in valid if p.id != parent.id]
        self.random_state.shuffle(pool)

        contexts: List[EvolvedProgram] = []
        seen_solutions = {parent.solution}
        for p in pool:
            if p.solution not in seen_solutions:
                contexts.append(p)
                seen_solutions.add(p.solution)
            if len(contexts) >= count:
                break

        return {"": parent}, {"": contexts}


# EVOLVE-BLOCK-END
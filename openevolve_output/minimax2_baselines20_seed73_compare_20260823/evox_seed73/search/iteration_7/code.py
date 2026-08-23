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
    """Adaptive elite search with parent rotation and diverse examples."""

    def __init__(self, name: str, config: DatabaseConfig):
        super().__init__(name, config)
        self.random_state = random.Random(getattr(config, "random_seed", None))
        self.initial_program = None
        self._best_score: Optional[float] = None
        self._last_meaningful_iteration = 0
        self._parent_uses: Dict[str, int] = {}
        self._label_uses: Dict[str, int] = {}
        self._sample_calls = 0

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

        current_iteration = iteration
        if not isinstance(current_iteration, int):
            current_iteration = program.iteration_found
        if not isinstance(current_iteration, int):
            current_iteration = 0

        if program.parent_id:
            self._parent_uses[program.parent_id] = (
                self._parent_uses.get(program.parent_id, 0) + 1
            )

        if isinstance(program.parent_info, tuple) and len(program.parent_info) >= 1:
            label = program.parent_info[0]
            if label:
                self._label_uses[label] = self._label_uses.get(label, 0) + 1

        score = self._score(program)
        if score is not None:
            if self._best_score is None:
                self._best_score = score
                self._last_meaningful_iteration = current_iteration
            elif score > self._best_score:
                absolute_gain = score - self._best_score
                relative_gain = absolute_gain / max(abs(self._best_score), 1e-12)
                if absolute_gain > 0.01 or relative_gain > 0.01:
                    self._last_meaningful_iteration = current_iteration
                self._best_score = score

        if iteration is not None:
            self.last_iteration = max(self.last_iteration, iteration)

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

        self._sample_calls += 1
        context_count = max(0, num_context_programs or 0)

        scored = [(self._score(p), p) for p in candidates]
        numeric = [(s, p) for s, p in scored if s is not None]

        if not numeric:
            parent = self.random_state.choice(candidates)
            contexts = [p for p in candidates if p.id != parent.id][:context_count]
            return {"": parent}, {"": contexts}

        best_score = max(score for score, _ in numeric)
        # Keep selection focused on genuinely strong constructions, while
        # rotating among equivalent elites rather than repeatedly mutating one.
        elite = [
            p for score, p in numeric
            if score >= best_score - max(0.002, abs(best_score) * 0.002)
        ]
        if not elite:
            elite = [p for _, p in numeric]

        self.random_state.shuffle(elite)
        elite.sort(
            key=lambda p: (
                self._parent_uses.get(p.id, 0),
                -(self._score(p) if self._score(p) is not None else -float("inf")),
            )
        )
        parent = elite[0]

        current_iteration = max(
            [p.iteration_found for p in candidates if isinstance(p.iteration_found, int)]
            or [0]
        )
        stagnating = current_iteration - self._last_meaningful_iteration >= 8

        # A single deliberate divergence is useful on a long plateau, but do
        # not repeatedly spend the short search window on labels that failed.
        can_diverge = (
            stagnating
            and self._label_uses.get(self.DIVERGE_LABEL, 0) == 0
            and self._sample_calls == 1
        )
        if can_diverge:
            return {self.DIVERGE_LABEL: parent}, {}

        # Context contains distinct solution texts: a near-best reference plus
        # nearby alternatives is more informative than repeated elite copies.
        others = [p for p in candidates if p.id != parent.id]
        self.random_state.shuffle(others)
        others.sort(
            key=lambda p: (
                -(self._score(p) if self._score(p) is not None else -float("inf")),
                self._parent_uses.get(p.id, 0),
            )
        )

        contexts: List[EvolvedProgram] = []
        seen_solutions = {parent.solution}
        for candidate in others:
            if candidate.solution in seen_solutions:
                continue
            contexts.append(candidate)
            seen_solutions.add(candidate.solution)
            if len(contexts) >= context_count:
                break

        return {"": parent}, {"": contexts}


# EVOLVE-BLOCK-END
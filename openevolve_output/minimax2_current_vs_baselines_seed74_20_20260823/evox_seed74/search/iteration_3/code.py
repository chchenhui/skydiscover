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
    def __init__(self, name: str, config: DatabaseConfig):
        super().__init__(name, config)
        self.random_state = random.Random(getattr(config, "random_seed", None))
        self.best_seen = float("-inf")
        self.last_meaningful_improvement = 0
        self.parent_uses: Dict[str, int] = {}
        self.diverge_uses = 0
        self.refine_uses = 0

    def add(self, program: EvolvedProgram, iteration: Optional[int] = None, **kwargs: Any) -> str:
        self.programs[program.id] = program
        step = iteration if iteration is not None else program.iteration_found
        if isinstance(step, int):
            self.last_iteration = max(getattr(self, "last_iteration", 0), step)

        if program.parent_id:
            self.parent_uses[program.parent_id] = self.parent_uses.get(program.parent_id, 0) + 1
        if program.parent_info and program.parent_info[0] == self.DIVERGE_LABEL:
            self.diverge_uses += 1
        if program.parent_info and program.parent_info[0] == self.REFINE_LABEL:
            self.refine_uses += 1

        score = program.metrics.get("combined_score")
        if isinstance(score, (int, float)) and math.isfinite(float(score)):
            score = float(score)
            meaningful = score > self.best_seen + max(0.01, abs(self.best_seen) * 0.01)
            if score > self.best_seen:
                self.best_seen = score
            if meaningful and isinstance(step, int):
                self.last_meaningful_improvement = step

        if self.config.db_path:
            self._save_program(program)
        self._update_best_program(program)
        return program.id

    def sample(
        self, num_context_programs: Optional[int] = 4, **kwargs: Any
    ) -> Tuple[Dict[str, EvolvedProgram], Dict[str, List[EvolvedProgram]]]:
        programs = list(self.programs.values())
        if not programs:
            raise ValueError("No candidates available for sampling")

        def score(p: EvolvedProgram) -> float:
            value = p.metrics.get("combined_score")
            return float(value) if isinstance(value, (int, float)) and math.isfinite(float(value)) else float("-inf")

        ranked = sorted(programs, key=score, reverse=True)
        top = ranked[:max(1, min(4, len(ranked)))]
        stalled = getattr(self, "last_iteration", 0) - self.last_meaningful_improvement >= 3

        # Prefer strong but underused parents rather than repeatedly selecting one elite.
        least_used = min(self.parent_uses.get(p.id, 0) for p in top)
        parent = self.random_state.choice(
            [p for p in top if self.parent_uses.get(p.id, 0) == least_used]
        )

        label = ""
        if stalled:
            # A plateau of identical elite scores needs a genuinely new construction.
            label = self.DIVERGE_LABEL if self.diverge_uses <= self.refine_uses else self.REFINE_LABEL

        if label:
            return {label: parent}, {}

        n = max(0, num_context_programs or 0)
        pool = [p for p in programs if p.id != parent.id]
        # Include both elite examples and the unusual low-score attempt when available.
        elite = [p for p in ranked if p.id != parent.id]
        unusual = list(reversed(ranked))
        self.random_state.shuffle(elite)
        self.random_state.shuffle(unusual)
        contexts: List[EvolvedProgram] = []
        for p in elite + unusual:
            if p.id not in [x.id for x in contexts]:
                contexts.append(p)
            if len(contexts) >= n:
                break

        return {"": parent}, {"": contexts[:n]}


# EVOLVE-BLOCK-END
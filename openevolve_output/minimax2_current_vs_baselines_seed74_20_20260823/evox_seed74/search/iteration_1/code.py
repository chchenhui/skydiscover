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
    """Adaptive elite search with parent-use balancing."""

    def __init__(self, name: str, config: DatabaseConfig):
        super().__init__(name, config)
        self.random_state = random.Random(getattr(config, "random_seed", None))
        self.best_seen: Optional[float] = None
        self.stagnant_steps = 0
        self.parent_uses: Dict[str, int] = {}
        self.label_uses: Dict[str, int] = {}

    def _score(self, program: EvolvedProgram) -> Optional[float]:
        value = program.metrics.get("combined_score")
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            score = float(value)
            if math.isfinite(score):
                return score
        return None

    def add(
        self, program: EvolvedProgram, iteration: Optional[int] = None, **kwargs: Any
    ) -> str:
        self.programs[program.id] = program

        if program.parent_id:
            self.parent_uses[program.parent_id] = (
                self.parent_uses.get(program.parent_id, 0) + 1
            )

        if program.parent_info and program.parent_info[0]:
            target_id = program.parent_info[1] or program.parent_id
            if target_id:
                self.label_uses[target_id] = self.label_uses.get(target_id, 0) + 1

        score = self._score(program)
        if score is not None:
            if self.best_seen is None:
                self.best_seen = score
            else:
                threshold = max(0.01, abs(self.best_seen) * 0.01)
                if score > self.best_seen + threshold:
                    self.best_seen = score
                    self.stagnant_steps = 0
                else:
                    self.stagnant_steps += 1

        if iteration is not None:
            self.last_iteration = max(getattr(self, "last_iteration", -1), iteration)

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
        numeric = [(s, p) for s, p in scored if s is not None]

        if numeric:
            best = max(s for s, _ in numeric)
            # Strongly exploit the elite set, while rotating among equally good
            # solutions instead of repeatedly mutating one ancestor.
            elite = [p for s, p in numeric if s >= best * 0.90]
            weights = [
                1.0 / (1.0 + self.parent_uses.get(p.id, 0))
                for p in elite
            ]
            parent = self.random_state.choices(elite, weights=weights, k=1)[0]
        else:
            parent = self.random_state.choice(candidates)

        others = [p for p in candidates if p.id != parent.id]
        self.random_state.shuffle(others)

        # Present good alternative constructions first, then contrasting ideas.
        others.sort(
            key=lambda p: (
                self._score(p) is None,
                -(self._score(p) if self._score(p) is not None else -float("inf")),
                self.parent_uses.get(p.id, 0),
            )
        )
        contexts = others[:count]

        label = ""
        # A short plateau warrants one focused refinement attempt per elite
        # program. Deep stalls later trigger a genuinely different direction.
        if self.stagnant_steps >= 4 and self.label_uses.get(parent.id, 0) == 0:
            label = self.DIVERGE_LABEL
        elif self.stagnant_steps >= 2 and self.label_uses.get(parent.id, 0) == 0:
            label = self.REFINE_LABEL

        if label:
            return {label: parent}, {"" : []}
        return {"": parent}, {"": contexts}


# EVOLVE-BLOCK-END
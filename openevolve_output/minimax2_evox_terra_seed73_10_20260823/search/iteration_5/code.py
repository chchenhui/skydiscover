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
    """Adaptive small-population search database."""

    def __init__(self, name: str, config: DatabaseConfig):
        super().__init__(name, config)
        self.initial_program = None
        self.random_state = random.Random(getattr(config, "random_seed", None))
        self.best_score_seen = float("-inf")
        self.stagnant_adds = 0
        self.parent_uses: Dict[str, int] = {}
        self.parent_gains: Dict[str, float] = {}

    def _score(self, program: EvolvedProgram) -> Optional[float]:
        value = program.metrics.get("combined_score")
        if isinstance(value, (int, float)):
            score = float(value)
            if math.isfinite(score):
                return score
        return None

    def add(
        self, program: EvolvedProgram, iteration: Optional[int] = None, **kwargs: Any
    ) -> str:
        if iteration == 0 or program.iteration_found == 0:
            self.initial_program = program

        score = self._score(program)
        previous_best = self.best_score_seen

        if score is not None:
            if previous_best == float("-inf"):
                self.best_score_seen = score
            else:
                meaningful = max(0.01, abs(previous_best) * 0.01)
                if score > previous_best + meaningful:
                    self.best_score_seen = score
                    self.stagnant_adds = 0
                else:
                    self.best_score_seen = max(self.best_score_seen, score)
                    self.stagnant_adds += 1

        if program.parent_id:
            parent = self.get(program.parent_id)
            self.parent_uses[program.parent_id] = self.parent_uses.get(program.parent_id, 0) + 1
            if parent is not None and score is not None:
                parent_score = self._score(parent)
                if parent_score is not None:
                    gain = score - parent_score
                    self.parent_gains[program.parent_id] = (
                        self.parent_gains.get(program.parent_id, 0.0) + gain
                    )

        self.programs[program.id] = program

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

        scored = [(p, self._score(p)) for p in candidates]
        valid = [(p, s) for p, s in scored if s is not None]

        if not valid:
            parent = self.random_state.choice(candidates)
            pool = [p for p in candidates if p.id != parent.id]
        else:
            best = max(s for _, s in valid)
            tolerance = max(0.01, abs(best) * 0.01)
            elites = [p for p, s in valid if s >= best - tolerance]

            # On a plateau, deliberately revisit a credible non-elite approach.
            # This is useful here because a very weak earlier lineage produced an
            # elite child, while repeated elite-only mutations have stalled.
            challengers = [
                p for p, s in valid
                if s < best - tolerance and s >= best * 0.70
            ]

            if self.stagnant_adds >= 4 and challengers:
                challenger = min(
                    challengers,
                    key=lambda p: (
                        self.parent_uses.get(p.id, 0),
                        -self.parent_gains.get(p.id, 0.0),
                        self.random_state.random(),
                    ),
                )
                parent = challenger
            else:
                parent = min(
                    elites,
                    key=lambda p: (
                        self.parent_uses.get(p.id, 0),
                        -self.parent_gains.get(p.id, 0.0),
                        self.random_state.random(),
                    ),
                )

            # Give the model several independent high-quality constructions,
            # rather than repeatedly supplying low-score context.
            pool = [p for p in elites if p.id != parent.id]
            if challengers and parent.id not in [p.id for p in challengers]:
                pool.extend([p for p in challengers if p.id != parent.id])

            seen = set()
            pool = [p for p in pool if not (p.id in seen or seen.add(p.id))]
            self.random_state.shuffle(pool)

        limit = max(0, num_context_programs or 0)
        context = pool[:limit]

        if len(context) < limit:
            extras = [p for p in candidates if p.id != parent.id and p.id not in {q.id for q in context}]
            self.random_state.shuffle(extras)
            context.extend(extras[: limit - len(context)])

        return {"": parent}, {"": context}


# EVOLVE-BLOCK-END
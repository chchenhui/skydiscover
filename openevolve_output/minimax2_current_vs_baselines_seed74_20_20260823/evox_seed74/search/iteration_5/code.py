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
    """Adaptive search strategy balancing useful parents and fresh directions."""

    def __init__(self, name: str, config: DatabaseConfig):
        super().__init__(name, config)
        self.initial_program = None
        self.random_state = random.Random(getattr(config, "random_seed", None))
        self.best_seen_score: Optional[float] = None
        self.stagnant_adds = 0
        self.parent_uses: Dict[str, int] = {}
        self.label_uses: Dict[str, int] = {}

    def _score(self, program: EvolvedProgram) -> Optional[float]:
        value = program.metrics.get("combined_score")
        if isinstance(value, (int, float)):
            return float(value)
        return None

    def add(
        self, program: EvolvedProgram, iteration: Optional[int] = None, **kwargs: Any
    ) -> str:
        """Store a program and record persistent progress/lineage signals."""
        if iteration == 0 or program.iteration_found == 0:
            self.initial_program = program

        self.programs[program.id] = program

        if program.parent_id:
            self.parent_uses[program.parent_id] = (
                self.parent_uses.get(program.parent_id, 0) + 1
            )

        parent_label = program.parent_info[0] if program.parent_info else ""
        if parent_label:
            self.label_uses[program.parent_id] = (
                self.label_uses.get(program.parent_id, 0) + 1
            )

        score = self._score(program)
        if score is not None:
            if self.best_seen_score is None:
                self.best_seen_score = score
            else:
                absolute_gain = score - self.best_seen_score
                relative_gain = absolute_gain / max(abs(self.best_seen_score), 1e-12)
                if absolute_gain > 0.01 or relative_gain > 0.01:
                    self.best_seen_score = score
                    self.stagnant_adds = 0
                else:
                    if score > self.best_seen_score:
                        self.best_seen_score = score
                    self.stagnant_adds += 1

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

        scored = [(self._score(p), p) for p in candidates]
        numeric = [(s, p) for s, p in scored if s is not None]
        numeric.sort(key=lambda pair: pair[0], reverse=True)

        # During a plateau, explicitly ask for a different construction.  Pick
        # among good but underused parents, rather than repeatedly showing the
        # same score-1.0 solution and its identical contexts.
        if self.stagnant_adds >= 3 and numeric:
            top_count = max(1, min(len(numeric), max(3, len(numeric) // 2)))
            top = [p for _, p in numeric[:top_count]]
            unused_label = [
                p for p in top if self.label_uses.get(p.id, 0) == 0
            ]
            pool = unused_label or top
            least_used = min(self.parent_uses.get(p.id, 0) for p in pool)
            pool = [p for p in pool if self.parent_uses.get(p.id, 0) == least_used]
            parent = self.random_state.choice(pool)
            return {self.DIVERGE_LABEL: parent}, {}

        # Otherwise favor strong candidates, while retaining some chance to
        # revisit a weaker program that may encode a genuinely different idea.
        if numeric:
            top_count = max(1, min(len(numeric), 4))
            top = [p for _, p in numeric[:top_count]]
            weights = [
                1.0 / (1 + self.parent_uses.get(p.id, 0))
                for p in top
            ]
            parent = self.random_state.choices(top, weights=weights, k=1)[0]
        else:
            parent = self.random_state.choice(candidates)

        context_limit = max(0, num_context_programs or 0)
        available = [p for p in candidates if p.id != parent.id]
        contexts: List[EvolvedProgram] = []

        # Give the model contrasting evidence: a strong example plus a notably
        # different-score example when they exist, then random fresh examples.
        if numeric:
            best = numeric[0][1]
            if best.id != parent.id:
                contexts.append(best)

            lowest = numeric[-1][1]
            if lowest.id != parent.id and lowest.id not in {p.id for p in contexts}:
                contexts.append(lowest)

        remaining = [p for p in available if p.id not in {c.id for c in contexts}]
        self.random_state.shuffle(remaining)
        contexts.extend(remaining[:max(0, context_limit - len(contexts))])

        return {"": parent}, {"": contexts[:context_limit]}


# EVOLVE-BLOCK-END
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
    """Adaptive elite search with parent-use balancing."""

    def __init__(self, name: str, config: DatabaseConfig):
        super().__init__(name, config)
        self.random_state = random.Random(getattr(config, "random_seed", None))
        self.best_score = float("-inf")
        self.last_meaningful_improvement = 0
        self.parent_uses: Dict[str, int] = {}
        self.label_uses: Dict[str, int] = {}
        self.added_count = 0

    def _score(self, program: EvolvedProgram) -> Optional[float]:
        value = program.metrics.get("combined_score")
        if isinstance(value, (int, float)):
            return float(value)
        return None

    def add(
        self, program: EvolvedProgram, iteration: Optional[int] = None, **kwargs: Any
    ) -> str:
        self.programs[program.id] = program
        self.added_count += 1

        if program.parent_id:
            self.parent_uses[program.parent_id] = self.parent_uses.get(program.parent_id, 0) + 1
        if program.parent_info and program.parent_info[0]:
            label = program.parent_info[0]
            self.label_uses[label] = self.label_uses.get(label, 0) + 1

        score = self._score(program)
        current_iteration = iteration if iteration is not None else program.iteration_found
        if score is not None:
            if self.best_score == float("-inf"):
                self.best_score = score
            else:
                absolute_gain = score - self.best_score
                relative_gain = absolute_gain / max(abs(self.best_score), 1e-12)
                if absolute_gain > 0 and (absolute_gain >= 0.01 or relative_gain >= 0.01):
                    self.best_score = score
                    self.last_meaningful_improvement = current_iteration
                elif score > self.best_score:
                    self.best_score = score

        if iteration is not None:
            self.last_iteration = max(self.last_iteration, iteration)
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

        scored = [(self._score(p), p) for p in candidates]
        valid = [(s, p) for s, p in scored if s is not None]

        if not valid:
            parent = self.random_state.choice(candidates)
            return {"": parent}, {"": []}

        best = max(score for score, _ in valid)
        # Treat near-best programs as equivalent elites, then rotate among them.
        elites = [p for score, p in valid if score >= best - max(0.01, abs(best) * 0.01)]
        least_used = min(self.parent_uses.get(p.id, 0) for p in elites)
        choices = [p for p in elites if self.parent_uses.get(p.id, 0) == least_used]
        parent = self.random_state.choice(choices)

        current_iteration = max(
            [self.last_iteration] + [p.iteration_found for p in candidates]
        )
        stagnant = current_iteration - self.last_meaningful_improvement >= 4

        # A long plateau among many tied elites calls for a focused improvement
        # attempt, while rotating targets prevents repeatedly refining one clone.
        if stagnant and len(elites) >= 2:
            refine_used = self.label_uses.get(self.REFINE_LABEL, 0)
            diverge_used = self.label_uses.get(self.DIVERGE_LABEL, 0)
            label = self.REFINE_LABEL if refine_used <= diverge_used else self.DIVERGE_LABEL
            return {label: parent}, {}

        # Otherwise show distinct elite implementations plus one contrasting
        # lower-scoring attempt, avoiding repeated parent/context duplication.
        others = [p for p in candidates if p.id != parent.id]
        elite_context = [p for p in others if p in elites]
        self.random_state.shuffle(elite_context)

        context: List[EvolvedProgram] = elite_context[: min(2, num_context_programs or 0)]
        remaining = [p for p in others if p.id not in {x.id for x in context}]
        lower = [p for p in remaining if self._score(p) is not None and self._score(p) < best]
        self.random_state.shuffle(lower)
        context.extend(lower[:1])

        if len(context) < (num_context_programs or 0):
            extras = [p for p in remaining if p.id not in {x.id for x in context}]
            self.random_state.shuffle(extras)
            context.extend(extras[: (num_context_programs or 0) - len(context)])

        return {"": parent}, {"": context}


# EVOLVE-BLOCK-END
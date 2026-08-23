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
    """Adaptive small-population search focused on strong, underused parents."""

    def __init__(self, name: str, config: DatabaseConfig):
        super().__init__(name, config)
        self.initial_program = None
        self.random_state = random.Random(getattr(config, "random_seed", None))

        # State is updated in add(), so it is reconstructed naturally as
        # programs are restored or replayed.
        self.best_score: Optional[float] = None
        self.best_history: List[float] = []
        self.no_progress_count = 0
        self.parent_uses: Dict[str, int] = {}
        self.context_uses: Dict[str, int] = {}
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
        """Store a program and update progress/lineage statistics."""
        if iteration == 0 or program.iteration_found == 0:
            self.initial_program = program

        previous_best = self.best_score
        score = self._score(program)

        if score is not None:
            if previous_best is None:
                self.best_score = score
                self.best_history.append(score)
            else:
                meaningful_delta = max(0.01, abs(previous_best) * 0.01)
                if score > previous_best + meaningful_delta:
                    self.best_score = score
                    self.best_history.append(score)
                    self.no_progress_count = 0
                else:
                    if score > previous_best:
                        self.best_score = score
                    self.no_progress_count += 1
        else:
            self.no_progress_count += 1

        if isinstance(program.parent_id, str) and program.parent_id:
            self.parent_uses[program.parent_id] = (
                self.parent_uses.get(program.parent_id, 0) + 1
            )

        if isinstance(program.other_context_ids, list):
            for context_id in program.other_context_ids:
                if isinstance(context_id, str) and context_id:
                    self.context_uses[context_id] = (
                        self.context_uses.get(context_id, 0) + 1
                    )

        if isinstance(program.parent_info, tuple) and len(program.parent_info) >= 1:
            label = program.parent_info[0]
            if isinstance(label, str) and label:
                self.label_uses[label] = self.label_uses.get(label, 0) + 1

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
        """Choose a high-quality but underused parent and useful peer context."""
        candidates = list(self.programs.values())
        if not candidates:
            raise ValueError("No candidates available for sampling")

        context_limit = max(0, num_context_programs or 0)
        scored = [(program, self._score(program)) for program in candidates]
        numeric = [(program, score) for program, score in scored if score is not None]

        if numeric:
            best = max(score for _, score in numeric)
            # With a tiny search budget, avoid spending generations mutating
            # clearly failed solutions. Keep a narrow near-best elite instead.
            tolerance = max(0.02, abs(best) * 0.05)
            elite = [program for program, score in numeric if score >= best - tolerance]
            if not elite:
                elite = [program for program, _ in numeric]
        else:
            elite = candidates[:]

        # Prefer elite candidates that have produced fewer children. Random
        # tie-breaking prevents repeatedly selecting the same equal-score program.
        minimum_uses = min(self.parent_uses.get(p.id, 0) for p in elite)
        parent_pool = [
            p for p in elite if self.parent_uses.get(p.id, 0) == minimum_uses
        ]
        parent = self.random_state.choice(parent_pool)

        # Repeated non-improvements indicate that a promising construction needs
        # a deliberate local revision. Labels are reserved for real stagnation.
        label = ""
        if self.no_progress_count >= 3:
            label = self.REFINE_LABEL

        if label:
            return {label: parent}, {}

        # Context is intentionally restricted to other strong programs. The
        # current population contains a severe low-score outlier, which is more
        # likely to distract than provide a useful construction reference.
        context_pool = [p for p in elite if p.id != parent.id]
        self.random_state.shuffle(context_pool)
        context_pool.sort(key=lambda p: self.context_uses.get(p.id, 0))

        contexts = context_pool[:context_limit]
        return {"": parent}, {"": contexts}


# EVOLVE-BLOCK-END
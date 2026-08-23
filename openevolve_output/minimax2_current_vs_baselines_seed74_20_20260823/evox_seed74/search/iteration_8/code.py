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
    """Adaptive database emphasizing historically productive lineages."""

    def __init__(self, name: str, config: DatabaseConfig):
        super().__init__(name, config)
        self.initial_program = None
        self.random_state = random.Random(getattr(config, "random_seed", None))
        self.best_score_seen = float("-inf")
        self.stagnation_count = 0
        self.add_count = 0

    def _score(self, program: EvolvedProgram) -> Optional[float]:
        value = program.metrics.get("combined_score")
        if isinstance(value, (int, float)) and not isinstance(value, bool):
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
        self.add_count += 1

        score = self._score(program)
        if score is not None:
            if self.best_score_seen == float("-inf"):
                self.best_score_seen = score
                self.stagnation_count = 0
            else:
                improvement = score - self.best_score_seen
                meaningful = improvement > 0.01 or improvement > 0.01 * abs(self.best_score_seen)
                if meaningful:
                    self.best_score_seen = score
                    self.stagnation_count = 0
                else:
                    self.best_score_seen = max(self.best_score_seen, score)
                    self.stagnation_count += 1

        if iteration is not None:
            self.last_iteration = max(self.last_iteration, iteration)

        if self.config.db_path:
            self._save_program(program)

        self._update_best_program(program)
        logger.debug("Added program %s", program.id)
        return program.id

    def sample(
        self, num_context_programs: Optional[int] = 4, **kwargs: Any
    ) -> Tuple[Dict[str, EvolvedProgram], Dict[str, List[EvolvedProgram]]]:
        candidates = list(self.programs.values())
        if not candidates:
            raise ValueError("No candidates available for sampling")

        count = num_context_programs or 0
        scores = {p.id: self._score(p) for p in candidates}

        # Measure which parent lineages actually generated stronger children.
        child_scores: Dict[str, List[float]] = {}
        parent_uses: Dict[str, int] = {}
        for child in candidates:
            if child.parent_id:
                parent_uses[child.parent_id] = parent_uses.get(child.parent_id, 0) + 1
                child_score = scores[child.id]
                if child_score is not None:
                    child_scores.setdefault(child.parent_id, []).append(child_score)

        productive = []
        for p in candidates:
            parent_score = scores[p.id]
            children = child_scores.get(p.id, [])
            if parent_score is not None and children:
                gain = max(children) - parent_score
                if gain > 0:
                    # Productive low-quality ancestors are valuable mutation prompts,
                    # but reduce repeated use of the exact same parent.
                    value = gain / math.sqrt(1 + parent_uses.get(p.id, 0))
                    productive.append((value, p))

        if productive:
            productive.sort(key=lambda item: item[0], reverse=True)
            shortlist = productive[:min(3, len(productive))]
            weights = [max(0.001, value) for value, _ in shortlist]
            parent = self.random_state.choices(
                [p for _, p in shortlist], weights=weights, k=1
            )[0]
        else:
            numeric = [p for p in candidates if scores[p.id] is not None]
            if numeric:
                numeric.sort(key=lambda p: scores[p.id])
                # Explore weak/underdeveloped approaches rather than repeatedly
                # mutating an already saturated elite.
                pool = numeric[:max(1, len(numeric) // 3)]
                parent = self.random_state.choice(pool)
            else:
                parent = self.random_state.choice(candidates)

        available = [p for p in candidates if p.id != parent.id]
        available.sort(
            key=lambda p: scores[p.id] if scores[p.id] is not None else float("-inf"),
            reverse=True,
        )

        # Elite context is useful here, but prefer distinct solution text so the
        # model receives multiple coordinate constructions rather than duplicates.
        elite = available[:max(count * 2, len(available) // 2, 1)]
        self.random_state.shuffle(elite)

        context: List[EvolvedProgram] = []
        seen_solutions = set()
        for p in elite:
            if len(context) >= count:
                break
            if p.solution not in seen_solutions:
                context.append(p)
                seen_solutions.add(p.solution)

        for p in elite + available:
            if len(context) >= count:
                break
            if p.id not in {chosen.id for chosen in context}:
                context.append(p)

        return {"": parent}, {"": context}


# EVOLVE-BLOCK-END
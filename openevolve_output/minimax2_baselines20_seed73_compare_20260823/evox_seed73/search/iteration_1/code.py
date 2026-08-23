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
    """Adaptive score-aware search database."""

    def __init__(self, name: str, config: DatabaseConfig):
        super().__init__(name, config)
        self.initial_program = None
        self.random_state = random.Random(getattr(config, "random_seed", None))
        self.best_score_seen = float("-inf")
        self.last_meaningful_iteration = 0
        self.add_count = 0
        self.parent_uses: Dict[str, int] = {}
        self.parent_gains: Dict[str, float] = {}
        self.best_history: List[float] = []

    def _score(self, program: EvolvedProgram) -> Optional[float]:
        value = program.metrics.get("combined_score")
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            score = float(value)
            if math.isfinite(score):
                return score
        return None

    def _meaningful_improvement(self, new_score: float, old_score: float) -> bool:
        gain = new_score - old_score
        if gain > 0.01:
            return True
        return old_score != 0 and gain / abs(old_score) > 0.01

    def add(
        self, program: EvolvedProgram, iteration: Optional[int] = None, **kwargs: Any
    ) -> str:
        if iteration == 0 or program.iteration_found == 0:
            self.initial_program = program

        self.programs[program.id] = program
        self.add_count += 1

        current_iteration = (
            iteration if iteration is not None else program.iteration_found
        )
        if isinstance(current_iteration, int):
            self.last_iteration = max(getattr(self, "last_iteration", 0), current_iteration)

        score = self._score(program)
        if score is not None:
            previous_best = self.best_score_seen
            if score > self.best_score_seen:
                self.best_score_seen = score
                if (
                    previous_best != float("-inf")
                    and self._meaningful_improvement(score, previous_best)
                ):
                    self.last_meaningful_iteration = current_iteration
            self.best_history.append(self.best_score_seen)

        # Record whether a selected parent produced a useful child. This state
        # allows sampling to favor productive lineages without repeatedly using
        # the same parent.
        if program.parent_id:
            parent_id = program.parent_id
            self.parent_uses[parent_id] = self.parent_uses.get(parent_id, 0) + 1
            parent = self.get(parent_id)
            parent_score = self._score(parent) if parent is not None else None
            if score is not None and parent_score is not None:
                gain = score - parent_score
                if gain > 0:
                    self.parent_gains[parent_id] = (
                        self.parent_gains.get(parent_id, 0.0) + gain
                    )

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

        context_count = max(0, num_context_programs or 0)
        scored = [(program, self._score(program)) for program in candidates]
        valid = [(program, score) for program, score in scored if score is not None]

        if not valid:
            parent = self.random_state.choice(candidates)
            others = [p for p in candidates if p.id != parent.id]
            self.random_state.shuffle(others)
            return {"": parent}, {"": others[:context_count]}

        valid.sort(key=lambda item: item[1], reverse=True)
        best_score = valid[0][1]
        current_iteration = getattr(self, "last_iteration", self.add_count)
        stalled = current_iteration - self.last_meaningful_iteration >= 5

        # During a plateau, explicitly revisit almost-best alternatives. For
        # geometric construction tasks these often differ from the optimum by
        # one coordinate or one missing symmetry, making them good mutation
        # targets when shown alongside an optimal example.
        near_best = [
            program
            for program, score in valid
            if score >= best_score * 0.97 and score < best_score - 1e-12
        ]
        elite_size = max(3, int(math.ceil(len(valid) * 0.45)))
        elite = [program for program, _ in valid[:elite_size]]

        if stalled and near_best and self.random_state.random() < 0.65:
            parent_pool = near_best
        else:
            parent_pool = elite

        def parent_weight(program: EvolvedProgram) -> float:
            uses = self.parent_uses.get(program.id, 0)
            gains = self.parent_gains.get(program.id, 0.0)
            score = self._score(program) or 0.0
            # High quality and previously productive parents are useful, but
            # inverse usage prevents a single lineage from monopolizing search.
            return (0.25 + score + 3.0 * gains) / (1.0 + uses)

        weights = [parent_weight(program) for program in parent_pool]
        parent = self.random_state.choices(parent_pool, weights=weights, k=1)[0]

        # Context starts with strong examples, then includes a near-best
        # alternative when available. This gives the LLM both a target and a
        # contrasting implementation rather than four redundant copies.
        context: List[EvolvedProgram] = []
        used_ids = {parent.id}

        def add_context(program: EvolvedProgram) -> None:
            if program.id not in used_ids and len(context) < context_count:
                context.append(program)
                used_ids.add(program.id)

        for program, _ in valid:
            add_context(program)
            if len(context) >= min(context_count, 2):
                break

        alternatives = [
            program for program in near_best if program.id not in used_ids
        ]
        self.random_state.shuffle(alternatives)
        for program in alternatives:
            add_context(program)

        remaining = [program for program, _ in valid if program.id not in used_ids]
        self.random_state.shuffle(remaining)
        for program in remaining:
            add_context(program)

        return {"": parent}, {"": context}


# EVOLVE-BLOCK-END
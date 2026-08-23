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
        self.best_seen = float("-inf")
        self.last_meaningful_improvement = 0
        self.parent_uses: Dict[str, int] = {}
        self.sample_count = 0

    def add(
        self, program: EvolvedProgram, iteration: Optional[int] = None, **kwargs: Any
    ) -> str:
        self.programs[program.id] = program

        if iteration is not None:
            self.last_iteration = max(self.last_iteration, iteration)

        score = program.metrics.get("combined_score")
        if isinstance(score, (int, float)):
            score = float(score)
            improvement = score - self.best_seen
            relative = improvement / max(abs(self.best_seen), 1e-9)
            if improvement > 0.01 or relative > 0.01:
                self.last_meaningful_improvement = (
                    iteration if iteration is not None else program.iteration_found
                )
            self.best_seen = max(self.best_seen, score)

        if program.parent_id:
            self.parent_uses[program.parent_id] = (
                self.parent_uses.get(program.parent_id, 0) + 1
            )

        if self.config.db_path:
            self._save_program(program)
        self._update_best_program(program)
        return program.id

    def sample(
        self, num_context_programs: Optional[int] = 4, **kwargs: Any
    ) -> Tuple[Dict[str, EvolvedProgram], Dict[str, List[EvolvedProgram]]]:
        scored = []
        for program in self.programs.values():
            score = program.metrics.get("combined_score")
            if isinstance(score, (int, float)):
                scored.append((float(score), program))

        if not scored:
            raise ValueError("No numeric candidates available for sampling")

        scored.sort(key=lambda item: item[0], reverse=True)
        self.sample_count += 1
        context_count = num_context_programs or 0

        # Keep a small elite pool, but rotate among it rather than repeatedly
        # mutating a single identical best program.
        elite_size = min(max(3, len(scored) // 4), len(scored))
        elite = [program for _, program in scored[:elite_size]]
        weights = [
            1.0 / (1.0 + self.parent_uses.get(program.id, 0))
            for program in elite
        ]
        parent = self.random_state.choices(elite, weights=weights, k=1)[0]

        current_iteration = max(
            self.last_iteration,
            max((p.iteration_found for _, p in scored), default=0),
        )
        stalled = current_iteration - self.last_meaningful_improvement >= 4

        # During a plateau, explicitly ask for a focused refinement of an elite
        # solution. Otherwise, use normal mutation with complementary examples.
        if stalled and self.sample_count % 2 == 1:
            return {self.REFINE_LABEL: parent}, {}

        pool = [program for _, program in scored if program.id != parent.id]
        contexts: List[EvolvedProgram] = []

        # Mostly near-best examples preserve useful geometric constructions.
        for program in pool[: max(0, context_count - 1)]:
            contexts.append(program)

        # One non-elite example can expose a genuinely different construction.
        if context_count and len(pool) > elite_size:
            alternatives = pool[elite_size:]
            candidate = self.random_state.choice(alternatives)
            if candidate.id not in {p.id for p in contexts}:
                contexts.append(candidate)

        for program in pool:
            if len(contexts) >= context_count:
                break
            if program.id not in {p.id for p in contexts}:
                contexts.append(program)

        return {"": parent}, {"": contexts[:context_count]}


# EVOLVE-BLOCK-END
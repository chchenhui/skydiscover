"""The strategy layer: what an expensive model contributes that a cheap one cannot.

EfficientEvolve splits generation in two. A *strategy* is a plan in prose --
which structural idea to apply and why it should score better -- and it is the
part that is genuinely hard to come up with, so it is bought from the
expensive model. Turning a plan into working code is normally bought in bulk
from the cheap model; sparse default insurance routes either a strategy-specific
or unguided implementation through the expensive tier so difficult code is not
left to the cheap model forever.

The economics only work if one strategy feeds many implementations, which makes
two things load-bearing: knowing when a strategy is spent, and telling the
expensive model what its previous strategies actually scored so it stops
re-proposing them.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

#: Tags used by strategy responses. Strategy calls deliberately request prose;
#: the controller's sparse code-insurance calls are a separate route.
TITLE_PATTERN = re.compile(r"<STRATEGY_TITLE>\s*(.*?)\s*</STRATEGY_TITLE>", re.I | re.S)
PLAN_PATTERN = re.compile(r"<STRATEGY>\s*(.*?)\s*</STRATEGY>", re.I | re.S)
REFERENCE_PATTERN = re.compile(
    r"<REFERENCE_IMPLEMENTATION>\s*(.*?)\s*</REFERENCE_IMPLEMENTATION>", re.I | re.S
)


@dataclass
class Strategy:
    """One plan, plus what it went on to achieve."""

    index: int
    title: str
    plan: str
    cost: float = 0.0
    #: Best score reached by any implementation of this strategy.
    best_score: Optional[float] = None
    #: Score the incumbent stood at when this strategy was proposed.
    baseline_score: Optional[float] = None
    #: Minimum score gain that counts as a real improvement rather than jitter.
    improvement_epsilon: float = 1e-9
    rounds: int = 0
    #: Implementation calls attempted, including parse/evaluation failures.
    attempts: int = 0
    #: Implementation calls that produced a usable evaluated program.
    implementations: int = 0
    #: Strong-model code attempts already spent on this plan.
    guide_implementation_attempts: int = 0
    #: Replays are new ledger entries, but share credit with the original
    #: strategy through this root index. ``None`` means this is the root.
    source_strategy_index: Optional[int] = None
    #: Positive global-incumbent gain actually caused by this strategy.  This
    #: differs from ``best_score - baseline_score`` because it sums useful
    #: gains across multiple activations without crediting unrelated hedges.
    cumulative_gain: float = 0.0
    #: Scale-free version of ``cumulative_gain`` used by the long-horizon
    #: scheduler to compare problems whose objective magnitudes differ.
    cumulative_relative_gain: float = 0.0
    productive_rounds: int = 0
    barren_rounds: int = 0
    #: Measured generation spend after the prose strategy has been bought.
    implementation_spend: float = 0.0
    last_used_iteration: int = -1
    failure_feedback: List[str] = field(default_factory=list)

    @property
    def improved(self) -> bool:
        if self.best_score is None or self.baseline_score is None:
            return False
        return self.best_score > self.baseline_score + self.improvement_epsilon

    @property
    def label(self) -> str:
        return f"#{self.index} {self.title}"

    @property
    def root_index(self) -> int:
        """Stable search-operator identity shared by replay entries."""
        return self.source_strategy_index or self.index

    def outcome(self) -> str:
        """One line for the next strategy prompt: what this plan actually did."""
        yield_text = f"; {self.implementations}/{self.attempts} implementations usable"
        if self.guide_implementation_attempts:
            yield_text += f"; {self.guide_implementation_attempts} guide implementation(s)"
        failures = (
            "; failures: " + " | ".join(self.failure_feedback[-3:]) if self.failure_feedback else ""
        )
        if self.best_score is None:
            return (
                f"produced nothing that evaluated ({self.implementations}/{self.attempts} usable)"
                f"{failures}"
            )
        if self.improved:
            gain = self.best_score - (self.baseline_score or 0.0)
            return (
                f"best score {self.best_score:.6f} (+{gain:.6f} — an improvement)"
                f"{yield_text}{failures}"
            )
        return (
            f"best score {self.best_score:.6f} (no improvement on {self.baseline_score:.6f})"
            f"{yield_text}{failures}"
        )

    def record_failure(self, feedback: str) -> None:
        """Keep a few distinct, compact failure reasons for the next guide."""
        feedback = re.sub(r"\s+", " ", feedback).strip()[:400]
        if feedback and feedback not in self.failure_feedback:
            self.failure_feedback.append(feedback)
            self.failure_feedback = self.failure_feedback[-3:]


@dataclass
class StrategyLedger:
    """Every strategy tried, in order, with its outcome."""

    entries: List[Strategy] = field(default_factory=list)

    def add(self, strategy: Strategy) -> None:
        self.entries.append(strategy)

    @property
    def current(self) -> Optional[Strategy]:
        return self.entries[-1] if self.entries else None

    def history_text(self, limit: int = 8) -> str:
        """Past strategies and their outcomes, for the strategy prompt.

        This is the whole reason to pay for an expensive model twice: without
        it the second call is a fresh draw from the same distribution and will
        propose the first idea again.
        """
        past = [s for s in self.entries if s.rounds > 0]
        if not past:
            return ""

        # A recent-only window eventually forgets the very operators that
        # produced the incumbent.  Keep a small hall of fame plus the most
        # recent failures.  This is deterministic and costs no extra guide
        # call; it only spends the bounded prompt slots more usefully.
        limit = max(1, int(limit))

        def gain(strategy: Strategy) -> float:
            if strategy.cumulative_relative_gain > 0:
                return strategy.cumulative_relative_gain
            if strategy.best_score is None or strategy.baseline_score is None:
                return 0.0
            return max(0.0, strategy.best_score - strategy.baseline_score)

        hall_size = max(1, limit // 3)
        hall = sorted(past, key=lambda strategy: (gain(strategy), strategy.index), reverse=True)
        hall = [strategy for strategy in hall if gain(strategy) > 0][:hall_size]
        selected = list(hall)
        selected_indices = {strategy.index for strategy in selected}
        for strategy in reversed(past):
            if strategy.index in selected_indices:
                continue
            selected.append(strategy)
            selected_indices.add(strategy.index)
            if len(selected) >= limit:
                break

        lines = []
        hall_indices = {strategy.index for strategy in hall}
        for strategy in selected:
            role = " [high-value operator]" if strategy.index in hall_indices else ""
            lines.append(f'- "{strategy.title}"{role} → {strategy.outcome()}')
        return (
            "\n\n# Strategies already tried\n\n"
            + "\n".join(lines)
            + "\n\nDo not restate any of these. If one of them improved the score, the next "
            "strategy may build on it, but it must add a new idea rather than repeat it. "
            "High-value operators are retained from the full run, not merely the recent window."
        )

    def cost_split(self) -> str:
        """``strategy=$x over n calls`` for the run log."""
        total = sum(s.cost for s in self.entries)
        return f"{len(self.entries)} strategies, {total:.6f} spent proposing them"


def parse_strategy(response: str, index: int) -> Optional[Strategy]:
    """Pull a titled plan out of a strategy response.

    Falls back to using the whole response as the plan when the model answers
    without tags: a usable plan in the wrong wrapper is still worth the call
    that produced it.
    """
    if not response or not response.strip():
        return None

    title_match = TITLE_PATTERN.search(response)
    plan_match = PLAN_PATTERN.search(response)

    plan = (plan_match.group(1) if plan_match else response).strip()
    if not plan:
        return None

    title = _strategy_title(title_match.group(1) if title_match else "", plan, index)

    return Strategy(index=index, title=title, plan=plan)


def _strategy_title(raw_title: str, plan: str, index: int) -> str:
    """Return a compact title even when a model emits malformed wrapper tags."""
    title = re.sub(r"</?STRATEGY(?:_TITLE)?[^>]*>", "", raw_title, flags=re.I).strip()
    if not title:
        title = re.split(r"(?<=[.!?])\s", plan.strip())[0]
        title = re.sub(r"</?STRATEGY(?:_TITLE)?[^>]*>", "", title, flags=re.I).strip()
    return title[:80].strip() or f"strategy {index}"


def parse_strategies(response: str, start_index: int, limit: int) -> List[Strategy]:
    """Parse up to ``limit`` repeated strategy blocks from one guide response.

    A single usable unwrapped plan remains valid for backward compatibility.
    """
    if not response or not response.strip() or limit < 1:
        return []

    titles = [value.strip() for value in TITLE_PATTERN.findall(response)]
    plans = [value.strip() for value in PLAN_PATTERN.findall(response) if value.strip()]
    if not plans:
        fallback = parse_strategy(response, start_index)
        return [fallback] if fallback is not None else []

    strategies: List[Strategy] = []
    for offset, plan in enumerate(plans[:limit]):
        index = start_index + offset
        raw_title = titles[offset] if offset < len(titles) else ""
        strategies.append(
            Strategy(index=index, title=_strategy_title(raw_title, plan, index), plan=plan)
        )
    return strategies


def parse_reference_implementation(response: str) -> Optional[str]:
    """Extract the optional guide-authored program kept outside prose plans."""
    match = REFERENCE_PATTERN.search(response or "")
    if not match or not match.group(1).strip():
        return None
    return match.group(1).strip()


def strategy_request(
    program: str,
    score: Optional[float],
    ledger: StrategyLedger,
    *,
    metrics: Optional[Dict[str, Any]] = None,
    target_hint: str = "",
    count: int = 1,
    include_reference: bool = False,
    history_limit: int = 8,
) -> str:
    """User message asking the expensive model for a small portfolio of plans."""
    score_line = f" It scores {score:.6f}." if score is not None else ""
    metric_lines = []
    for name, value in (metrics or {}).items():
        if isinstance(value, bool):
            rendered = str(value)
        elif isinstance(value, (int, float)):
            rendered = f"{float(value):.12g}"
        elif isinstance(value, str) and len(value) <= 120:
            rendered = value.replace("\n", " ")
        else:
            continue
        metric_lines.append(f"- {name}: {rendered}")
    metrics_text = (
        "\n\n# Current evaluated metrics\n\n"
        + "\n".join(metric_lines)
        + "\n\nThe controller ranks candidates by the scalar score shown above "
        "(`combined_score` when that metric is present). Metric names denote "
        "different quantities and may use different units or scales."
        if metric_lines
        else ""
    )
    count = max(1, int(count))
    quantity = "exactly ONE strategy" if count == 1 else f"exactly {count} distinct strategies"
    diversity = (
        ""
        if count == 1
        else (
            "\nThe strategies must use genuinely different constructions or mechanisms, "
            "not parameter variations of one idea. Rank the immediately executable, hard-constraint-"
            "preserving strategy with the highest predicted combined score FIRST; it is activated "
            "first and may also receive the executable reference. Use implementation reliability "
            "to break near-ties, not to put a tiny safe gain ahead of a credible structural gain. "
            "Across a portfolio of three or more, cover "
            "an explicit local exploitation, a mesoscopic subsystem replacement, and a "
            "high-upside structural exploration, in whichever order their evidence warrants. "
            "For two strategies, cover local and structural roles. Do not fill the portfolio "
            "with only tiny perturbations or only speculative reconstructions."
        )
    )
    reference_contract = (
        """

After all strategy blocks, also implement the FIRST strategy once as a complete,
standalone program. This is an executable reference for the cheap implementers,
not an additional strategy. Preserve the required public function/signature and
wrap exactly one complete program as follows:

Because this program is evaluated as a candidate, strategy 1 and its reference
must be the valid, immediately executable proposal with the highest predicted
combined score. Use reliability only to break near-ties. Do not make it a
conservative local proposal merely to satisfy portfolio diversity; the remaining
strategies can cover other scales.

<REFERENCE_IMPLEMENTATION>
```python
the complete program
```
</REFERENCE_IMPLEMENTATION>

The program must be immediately executable and satisfy every hard constraint.
Do not put code inside any <STRATEGY> block.
"""
        if include_reference
        else ""
    )
    return f"""# Current parent program{score_line}

```
{program}
```{metrics_text}
{ledger.history_text(history_limit)}{target_hint}

# Your task

Propose {quantity} for improving this program's score.{diversity}

A strategy is a **plan, not code**. Say which structural idea to apply and why it
should score better than what is there now. Be concrete enough that a competent
implementer needs to make no further design decisions — name the construction,
the arrangement, the invariant, whatever the idea turns on. Do not write the
program: someone else implements it.

Every strategy must be self-contained and immediately implementable from the
program and context above. Do not depend on an offline solve, an unavailable
certificate or lookup table, fabricated constants/data, external tools, or a
future optimization step. State how the implementation preserves the problem's
hard validity constraints. Reject your own idea if you cannot provide all of the
information the implementer needs. An "intended", "designed", "about", or
hypothetical objective value is not a gain estimate unless it is derived from
the explicit constants or local transformation supplied in the strategy.

Before returning an idea, reject it if its calculable estimate or bound does not
strictly exceed the incumbent. Compare like with like: a predicted raw objective
must be compared with the same named incumbent metric, never with
`combined_score`. The expected-gain sentence must end by giving the predicted
`combined_score` and showing that it strictly exceeds the current ranking score;
if conversion is needed, state it. If an exact prediction is impossible, instead
identify one precise, checkable local mechanism that strictly raises the same
objective while preserving every factor used by `combined_score` and a valid
incumbent fallback. Do not expand this one-sentence check into a proof.

Within each requested portfolio role, choose the idea with the best balance of
calculable gain and implementation reliability. Structural changes are useful
early, but once the incumbent is strong, the portfolio must preserve both a
local, verifiable refinement path and a distinct route that can escape the
incumbent's structural ceiling.

When the program optimizes a continuous objective built from minima, maxima,
ratios, or other nonsmooth extrema, explicitly test whether an auxiliary
objective variable can turn it into smooth inequality constraints. If
applicable, include a deterministic multistart constrained-optimization route
with analytic constraint derivatives where practical; do not make every route
another random perturbation of the incumbent. Ignore this instruction when the
task is not a continuous numerical optimization problem.

Respond with the following block once per strategy, for exactly {count} block(s):

<STRATEGY_TITLE>a short name, a few words</STRATEGY_TITLE>
<STRATEGY>
the plan, including its one-sentence expected-gain check, 3-8 sentences
</STRATEGY>
{reference_contract}
"""


def implementation_instruction(strategy: Strategy, variant: int, total: int) -> str:
    """Appended to the normal generation prompt for the cheap model."""
    variant_line = (
        ""
        if total <= 1
        else (
            f"\n\nThis is variant {variant} of {total}. The other variants implement the same "
            "strategy, so differ from them only in how you resolve the choices the plan leaves "
            "open — not by substituting a different idea."
        )
    )
    return f"""

# Strategy to implement: {strategy.title}

{strategy.plan}

Implement exactly this strategy. Do not substitute your own idea, and do not fall
back to a small tweak of the current program: the plan above is what is being
tested. If you return SEARCH/REPLACE edits, copy every SEARCH block byte-for-byte
from the current parent program and verify that it occurs there exactly once;
prefer replacing one complete small function or block over reconstructing several
fragile line fragments. An unmatched SEARCH block produces no candidate. If the
plan is ambiguous, pick the reading that maximises the objective and follow it
through.{variant_line}
"""

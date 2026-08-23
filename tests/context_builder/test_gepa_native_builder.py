"""Regression tests for GEPA-native prompt rendering."""

from skydiscover.config import Config
from skydiscover.context_builder.gepa_native import GEPANativeContextBuilder
from skydiscover.search.base_database import Program


def _program(program_id: str, solution: str, **kwargs) -> Program:
    return Program(
        id=program_id,
        solution=solution,
        metrics=kwargs.pop("metrics", {"combined_score": 0.5}),
        **kwargs,
    )


def test_diff_prompt_preserves_authoritative_solution_whitespace():
    solution = "def solve():\n    value = 1\n\n\n\n    return value\n"
    builder = GEPANativeContextBuilder(
        Config.from_dict({"language": "python", "diff_based_generation": True})
    )

    prompt = builder.build_prompt(
        _program("parent", solution),
        {"program_metrics": {"combined_score": 0.5}},
    )["user"]

    assert f"```python\n{solution}\n```" in prompt
    assert prompt.count("# Current Solution\n") == 1
    assert "<!-- BEGIN AUTHORITATIVE CURRENT SOLUTION -->" in prompt


def test_rejected_attempts_appear_before_authoritative_solution():
    parent = _program("parent", "def solve():\n    return 1")
    rejected = _program(
        "rejected",
        "def solve():\n    return 0",
        parent_id="parent",
        metadata={"changes": "Returned the wrong value"},
    )
    builder = GEPANativeContextBuilder(
        Config.from_dict({"language": "python", "diff_based_generation": True})
    )

    prompt = builder.build_prompt(
        parent,
        {
            "program_metrics": parent.metrics,
            "rejection_history": [rejected],
            "rejection_parent_scores": {"parent": 0.5},
        },
    )["user"]

    assert prompt.index("### Recent Rejected Attempts") < prompt.index(
        "<!-- BEGIN AUTHORITATIVE CURRENT SOLUTION -->"
    )
    assert "Never copy SEARCH text from previous attempts" in prompt


def test_prompt_optimization_has_one_current_prompt_heading():
    builder = GEPANativeContextBuilder(
        Config.from_dict({"language": "text", "diff_based_generation": False})
    )

    prompt = builder.build_prompt(
        _program("parent", "Answer with valid JSON."),
        {"program_metrics": {"combined_score": 0.5}},
    )["user"]

    assert prompt.count("# Current Prompt\n") == 1
    assert "# Current Solution\n" not in prompt
    assert "<!-- BEGIN AUTHORITATIVE CURRENT PROMPT -->" in prompt

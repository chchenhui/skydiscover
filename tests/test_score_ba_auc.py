"""Regression tests for offline BA-AUC reconstruction."""

import json

import pytest

from scripts.score_ba_auc import _chronological_log_samples, _resolve_program, reconstruct


def _write_program(path, program_id, score, parent_id):
    path.write_text(
        json.dumps(
            {
                "id": program_id,
                "parent_id": parent_id,
                "metrics": {"combined_score": score},
            }
        )
    )


def test_native_reconstruction_replays_overhead_and_candidates_in_log_order(tmp_path):
    run_dir = tmp_path / "run"
    programs_dir = run_dir / "checkpoints" / "checkpoint_2" / "programs"
    logs_dir = run_dir / "logs"
    programs_dir.mkdir(parents=True)
    logs_dir.mkdir()

    seed = "00000000-0000-0000-0000-000000000001"
    first = "00000000-0000-0000-0000-000000000002"
    second = "00000000-0000-0000-0000-000000000003"
    _write_program(programs_dir / "seed.json", seed, 0.2, None)
    _write_program(programs_dir / "first.json", first, 0.7, seed)
    _write_program(programs_dir / "second.json", second, 0.5, first)

    # The second call models controller overhead: it costs money but does not
    # directly produce a solution.  The final call models trailing failed work.
    (logs_dir / "run.log").write_text(
        "\n".join(
            [
                "LLM call openai/model: total_tokens=10, total_cost=$0.1000",
                "LLM call openai/model: total_tokens=20, total_cost=$0.2000",
                f"Iteration 1: Program {first} (parent: {seed}) completed",
                "LLM call openai/model: total_tokens=30, total_cost=$0.4000",
                f"Iteration 2: Program {second} (parent: {first}) completed",
                "LLM call openai/model: total_tokens=40, total_cost=$0.5000",
            ]
        )
    )

    samples = reconstruct(str(run_dir))
    assert [cost for cost, _ in samples] == pytest.approx([0.0, 0.3, 0.7, 1.2])
    assert [score for _, score in samples] == pytest.approx([0.2, 0.7, 0.5, 0.0])


def test_short_program_id_is_resolved_only_when_unique():
    first = {"id": "12345678-aaaa-aaaa-aaaa-aaaaaaaaaaaa", "metrics": {}}
    second = {"id": "87654321-bbbb-bbbb-bbbb-bbbbbbbbbbbb", "metrics": {}}
    programs = {first["id"]: first, second["id"]: second}

    assert _resolve_program(programs, "12345678") is first
    programs["12345678-cccc-cccc-cccc-cccccccccccc"] = {"id": "duplicate"}
    assert _resolve_program(programs, "12345678") is None


def test_chronological_log_samples_accepts_ada_short_program_id():
    program_id = "12345678-aaaa-aaaa-aaaa-aaaaaaaaaaaa"
    programs = {
        "seed": {"id": "seed", "parent_id": None, "metrics": {"combined_score": 0.1}},
        program_id: {
            "id": program_id,
            "parent_id": "seed",
            "metrics": {"combined_score": 0.9},
        },
    }
    text = "\n".join(
        [
            "LLM call model: total_tokens=10, total_cost=$0.0500",
            "Iteration 1: Program 12345678 (parent: seed) completed in 1.0s",
        ]
    )

    assert _chronological_log_samples(text, programs) == [(0.0, 0.1), (0.05, 0.9)]

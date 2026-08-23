"""Logging behavior shared by native search controllers."""

import logging
from types import SimpleNamespace

from skydiscover.search.base_database import Program
from skydiscover.search.default_discovery_controller import DiscoveryController
from skydiscover.search.utils.discovery_utils import SerializableResult


def test_candidate_metrics_are_logged_at_info(caplog):
    controller = object.__new__(DiscoveryController)
    controller.config = SimpleNamespace(diff_based_generation=False, checkpoint_interval=10)
    controller.monitor_callback = None

    class FakeDatabase:
        best_program_id = None

        def add(self, program, iteration=None):
            self.best_program_id = program.id

        def log_prompt(self, **kwargs):
            pass

    controller.database = FakeDatabase()
    child = Program(
        id="candidate",
        solution="pass",
        metrics={"validity": 1.0, "combined_score": 0.75},
    )
    result = SerializableResult(child_program_dict=child.to_dict())

    with caplog.at_level(
        logging.INFO,
        logger="skydiscover.search.default_discovery_controller",
    ):
        controller._process_iteration_result(result, iteration=1)

    assert "Metrics: validity=1.0000, combined_score=0.7500" in caplog.text

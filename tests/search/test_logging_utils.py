"""Tests for per-run logging level configuration."""

import logging

from skydiscover.search.utils.logging_utils import setup_search_logging


def test_info_log_file_excludes_third_party_debug(tmp_path):
    root = logging.getLogger()
    package_logger = logging.getLogger("skydiscover")
    original_handlers = root.handlers[:]
    original_level = root.level
    original_package_level = package_logger.level
    root.handlers = []
    package_logger.setLevel(logging.WARNING)

    try:
        setup_search_logging("INFO", str(tmp_path), "test")
        third_party = logging.getLogger("third_party_test_logger")
        third_party.debug("secret debug payload")
        third_party.info("visible info message")

        file_handlers = [h for h in root.handlers if isinstance(h, logging.FileHandler)]
        assert len(file_handlers) == 1
        file_handlers[0].flush()
        log_contents = (tmp_path / file_handlers[0].baseFilename.split("/")[-1]).read_text()

        assert file_handlers[0].level == logging.INFO
        assert "visible info message" in log_contents
        assert "secret debug payload" not in log_contents
        assert package_logger.level == logging.INFO
    finally:
        for handler in root.handlers:
            if handler not in original_handlers:
                handler.close()
        root.handlers = original_handlers
        root.setLevel(original_level)
        package_logger.setLevel(original_package_level)


def test_a_second_run_stops_writing_into_the_first_runs_log(tmp_path):
    """One process, several searches: each log must hold only its own run.

    Without retiring the previous file handler the first log becomes a
    superset of the whole session, which silently corrupts per-run analysis.
    """
    import logging

    from skydiscover.search.utils.logging_utils import setup_search_logging

    root = logging.getLogger()
    saved = list(root.handlers)
    try:
        setup_search_logging("INFO", str(tmp_path / "a"), "run_a")
        logging.getLogger("skydiscover.test").info("belongs to run a")

        setup_search_logging("INFO", str(tmp_path / "b"), "run_b")
        logging.getLogger("skydiscover.test").info("belongs to run b")

        first = next((tmp_path / "a").glob("*.log")).read_text()
        second = next((tmp_path / "b").glob("*.log")).read_text()

        assert "belongs to run a" in first
        assert "belongs to run b" not in first
        assert "belongs to run b" in second
    finally:
        for handler in list(root.handlers):
            if handler not in saved:
                root.removeHandler(handler)
                handler.close()
        for handler in saved:
            if handler not in root.handlers:
                root.addHandler(handler)

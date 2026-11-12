"""Shared test fixtures."""

import logging

import pytest


@pytest.fixture(autouse=True)
def _restore_classifier_logger():
    """The widget configures the global "classifier" logger (napari handler,
    no propagation). Restore it after each test so logging-related assertions
    (caplog) in other tests aren't affected by test order."""
    logger = logging.getLogger("classifier")
    state = (list(logger.handlers), logger.propagate, logger.level, logger.disabled)
    yield
    logger.handlers[:], logger.propagate, logger.level, logger.disabled = (
        state[0],
        state[1],
        state[2],
        state[3],
    )

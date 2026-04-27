"""Shared test factories for config construction."""

from pyaugmecon.config import PyAugmeconConfig


def make_config(name: str, **overrides: object) -> PyAugmeconConfig:
    """Build a PyAugmeconConfig with sensible test defaults.

    Disables CSV output, progress bars, and console logging by default.
    Any keyword can be overridden (e.g. ``workers``, ``work_distribution``).
    """
    defaults: dict[str, object] = {
        "workers": 1,
        "write_csv": False,
        "progress_bar": False,
        "log_to_console": False,
    }
    return PyAugmeconConfig(name=name, **(defaults | overrides))

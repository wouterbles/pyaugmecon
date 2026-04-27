"""Logging setup for PyAugmecon runs."""

from __future__ import annotations

import logging as std_logging
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any

from loguru import logger as log

if TYPE_CHECKING:
    from pyaugmecon.config import PyAugmeconConfig
    from pyaugmecon.results import PyAugmeconResult


def configure_loguru(
    logfile: str,
    log_to_console: bool,
    log_sink: Any | None = None,
) -> None:
    """Set up Loguru file sink and optional console sink for one run."""
    # Silence noisy Pyomo logger before adding our own sinks.
    std_logging.getLogger("pyomo.core").setLevel(std_logging.ERROR)

    log.remove()
    Path(logfile).parent.mkdir(parents=True, exist_ok=True)

    log.add(
        logfile,
        level="INFO",
        format="[{time:YYYY-MM-DD HH:mm:ss}] {message}",
        enqueue=True,
    )
    if log_to_console:
        log.add(sys.stdout, level="INFO", format="{message}")
    if log_sink is not None:
        log.add(
            lambda message: log_sink.info(message.record["message"]),
            level="INFO",
            format="{message}",
        )


def log_row(label: str, items: Sequence[tuple[str, str]]) -> None:
    """Log a single line: `label: k=v · k=v · k=v`."""
    body = " · ".join(f"{k}={v}" for k, v in items)
    log.info(f"  {label:<10} {body}")


def log_run_header(config: PyAugmeconConfig, version: str, artifact_dir: str) -> None:
    """Log the run banner and configuration summary at solve start."""
    solver_summary = config.solver_name
    if config.solver_io is not None:
        solver_summary += f" ({config.solver_io})"
    if config.solver_options:
        opts = ", ".join(f"{k}={v}" for k, v in config.solver_options.items())
        solver_summary += f" [{opts}]"

    if config.mode == "exact":
        grid = "from payoff ranges"
    else:
        grid = f"sampled ({config.sample_points})"
    if config.nadir_points is not None:
        grid += ", explicit nadir"

    skips_off = [
        name
        for name, on in (
            ("early_exit", config.early_exit),
            ("bypass", config.bypass),
            ("flag", config.flag),
        )
        if not on
    ]

    log.info(f"PyAugmecon v{version} · {config.name} ({config.mode})")
    log_row("Grid", [("points", grid)])
    log_row(
        "Parallel",
        [
            ("workers", str(config.workers)),
            ("work", config.work_distribution),
            ("flags", config.flag_policy),
        ],
    )
    log_row("Solver", [("backend", solver_summary)])
    if skips_off:
        log_row("Skips", [("disabled", ", ".join(skips_off))])
    log_row("Output", [("dir", artifact_dir)])


def log_run_summary(result: PyAugmeconResult) -> None:
    """Log the runtime and solver counts at solve end."""
    log.info(f"Done in {result.runtime_seconds:.2f}s")
    log_row("Pareto", [("solutions", str(result.count))])
    log_row(
        "Solver",
        [
            ("solved", str(result.models_solved)),
            ("infeasible", str(result.models_infeasible)),
            ("skipped", str(result.skipped_points)),
        ],
    )

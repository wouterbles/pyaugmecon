"""Benchmark cases (MOMKP instances) and method/parallelization scenarios.

Reproduces the 7 instances and 8 scenarios from the parallelized AUGMECON-R
paper, using current PyAugmecon config field names.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field

from pyaugmecon.example_models import kp_model


@dataclass(frozen=True, slots=True)
class BenchmarkCase:
    """One MOMKP instance: model factory plus grid and nadir overrides."""

    name: str
    objective_count: int
    grid_points: int
    nadir_points: list[int] | None = None

    def build_model(self):
        return kp_model(self.name, self.objective_count)

    def case_opts(self) -> dict[str, object]:
        """PyAugmecon kwargs for grid resolution and (optionally) nadirs."""
        opts: dict[str, object] = {"sample_points": self.grid_points}
        if self.nadir_points is not None:
            opts["nadir_points"] = list(self.nadir_points)
        return opts


@dataclass(frozen=True, slots=True)
class Scenario:
    """Method / parallelization configuration applied on top of a case."""

    name: str
    description: str
    opts: dict[str, object] = field(default_factory=dict)


# Grid sizes and nadirs taken from the original `benchmarks/model_data.py`.
BENCHMARK_CASES: dict[str, BenchmarkCase] = {
    c.name: c
    for c in [
        BenchmarkCase("2kp50", 2, 492),
        BenchmarkCase("2kp100", 2, 823),
        BenchmarkCase("2kp250", 2, 2534),
        BenchmarkCase("3kp40", 3, 540, [1031, 1069]),
        BenchmarkCase("3kp50", 3, 847, [1124, 1041]),
        BenchmarkCase("4kp40", 4, 141, [138, 106, 121]),
        BenchmarkCase("4kp50", 4, 53, [718, 717, 705]),
    ]
}

# Shared opt dicts to reduce repetition across scenarios.
_SINGLE = {"workers": 1, "work_distribution": "fixed", "flag_policy": "local"}
_R = {"flag": True, "bypass": True, "early_exit": True}
# `process_timeout=4h` matches the original benchmarks for the slowest
# serial methods; AUGMECON-R variants finish well under the default.
_FOUR_HOURS = 3600 * 4

BENCHMARK_SCENARIOS: dict[str, Scenario] = {
    s.name: s
    for s in [
        Scenario(
            "augmecon",
            "Plain AUGMECON: no flag, no bypass, no redivide, single process.",
            {
                **_SINGLE,
                "flag": False,
                "bypass": False,
                "early_exit": False,
                "process_timeout": _FOUR_HOURS,
            },
        ),
        Scenario(
            "augmecon_2",
            "AUGMECON2: bypass on, flag off, single process.",
            {
                **_SINGLE,
                "flag": False,
                "bypass": True,
                "early_exit": False,
                "process_timeout": _FOUR_HOURS,
            },
        ),
        Scenario(
            "augmecon_r",
            "AUGMECON-R: flag + bypass, single process.",
            {**_SINGLE, **_R},
        ),
        Scenario(
            "parallel_default",
            "Parallel AUGMECON-R: redivide work, shared flag.",
            {"work_distribution": "dynamic", "flag_policy": "shared", **_R},
        ),
        Scenario(
            "parallel_simple",
            "Parallel: no redivide, no shared flag.",
            {"work_distribution": "fixed", "flag_policy": "local", **_R},
        ),
        Scenario(
            "parallel_no_redivide",
            "Parallel: no redivide, shared flag.",
            {"work_distribution": "fixed", "flag_policy": "shared", **_R},
        ),
        Scenario(
            "parallel_no_shared_flag",
            "Parallel: redivide work, local flag.",
            {"work_distribution": "dynamic", "flag_policy": "local", **_R},
        ),
        Scenario(
            "parallel_outer_grid",
            "Parallel: outer-grid distribution, shared flag (current default for exact mode).",
            {"work_distribution": "outer_grid", "flag_policy": "shared", **_R},
        ),
    ]
}


def parse_names(spec: str, registry: Mapping[str, object], kind: str) -> list[str]:
    """Resolve `'all'` or a comma-separated selection against `registry`."""
    if spec.strip().lower() == "all":
        return list(registry)
    selected = [s.strip() for s in spec.split(",") if s.strip()]
    if not selected:
        raise ValueError(f"At least one {kind} must be selected.")
    unknown = [n for n in selected if n not in registry]
    if unknown:
        raise ValueError(
            f"Unknown {kind}(s): {', '.join(unknown)}. "
            f"Available: {', '.join(registry)}."
        )
    return selected

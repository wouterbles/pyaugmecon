"""Simple Pydantic configuration for PyAugmecon."""

from __future__ import annotations

from multiprocessing import cpu_count
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

Mode = Literal["exact", "sampled"]
ObjectiveOrder = Literal["auto_range", "given"]
WorkDistribution = Literal["auto", "dynamic", "fixed", "outer_grid"]
FlagPolicy = Literal["auto", "local", "shared"]
NadirStrategy = Literal["safe", "payoff"]


class PyAugmeconConfig(BaseModel):
    """Pydantic model for all user-provided runtime configuration."""

    model_config = ConfigDict(extra="forbid")

    name: str = ""
    mode: Mode = "exact"
    # Accepts int | list[int] on input; normalized to list[int] | None by validator.
    sample_points: list[int] | None = None
    nadir_points: list[int | float] | None = None
    nadir_strategy: NadirStrategy = "safe"
    nadir_undercut: float = Field(default=0.8, gt=0, le=1)
    objective_order: ObjectiveOrder = "auto_range"

    workers: int = Field(default_factory=cpu_count, ge=1)
    work_distribution: WorkDistribution = "auto"
    flag_policy: FlagPolicy = "auto"
    process_timeout: float | None = Field(default=None, gt=0)

    solve_warmstart: bool = True
    store_decision_variables: bool = False

    early_exit: bool = True
    bypass: bool = True
    flag: bool = True

    objective_tolerance: float = Field(default=1e-6, gt=0)
    penalty_weight: float = Field(default=1e-3, gt=0)
    round_decimals: int = Field(default=9, ge=0)

    artifact_folder: str = "logs"
    artifact_name: str | None = None
    write_csv: bool = True
    process_logging: bool = False
    progress_bar: bool = True
    log_to_console: bool = True

    solver_name: str = "highs"
    solver_io: str | None = None
    solver_options: dict[str, Any] = Field(default_factory=dict)

    @field_validator("sample_points", mode="before")
    @classmethod
    def _validate_sample_points(cls, value: object) -> list[int] | None:
        """Normalize scalar sample_points to a single-element list; validate all entries >= 2."""
        if value is None:
            return None
        if isinstance(value, int):
            value = [value]
        if not isinstance(value, list) or not value:
            raise TypeError("`sample_points` must be a non-empty int or list of ints.")
        result: list[int] = []
        for entry in value:
            if not isinstance(entry, int) or entry < 2:
                raise ValueError("`sample_points` entries must be integers >= 2.")
            result.append(entry)
        return result

    @field_validator("artifact_folder")
    @classmethod
    def _validate_artifact_folder(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("`artifact_folder` must not be empty.")
        return value

    @field_validator("artifact_name")
    @classmethod
    def _validate_artifact_name(cls, value: str | None) -> str | None:
        if value is None:
            return None
        value = value.strip()
        if not value:
            raise ValueError("`artifact_name` must not be blank when provided.")
        return value

    @field_validator("solver_options")
    @classmethod
    def _drop_none_solver_options(cls, value: dict[str, Any]) -> dict[str, Any]:
        return {key: option for key, option in value.items() if option is not None}

    @model_validator(mode="after")
    def _apply_defaults(self) -> PyAugmeconConfig:
        """Validate cross-field invariants and fill in mode-dependent defaults.

        The `auto` parallel choices become concrete values here. Exact
        multi-worker runs use outer-grid work plus a shared flag policy because
        that lets workers share skip information. Sampled and single-worker
        runs use simpler dynamic work plus local flags because the shared state
        has little chance to pay for itself there.
        """
        if not self.name:
            self.name = f"{self.solver_name}-{self.mode}"

        if self.mode == "sampled" and self.sample_points is None:
            raise ValueError("`sample_points` is required in sampled mode.")
        if self.mode == "exact" and self.sample_points is not None:
            raise ValueError("`sample_points` is only valid in sampled mode.")

        if self.work_distribution == "auto":
            self.work_distribution = (
                "outer_grid" if self.mode == "exact" and self.workers > 1 else "dynamic"
            )

        if self.flag_policy == "auto":
            self.flag_policy = (
                "shared" if self.mode == "exact" and self.workers > 1 else "local"
            )

        return self

    def validate_against_model(self, num_objectives: int) -> None:
        """Validate configuration rules that depend on the user model."""
        if num_objectives < 2:
            raise ValueError("At least two objectives are required in `obj_list`.")

        constrained_count = num_objectives - 1
        if (
            self.nadir_points is not None
            and len(self.nadir_points) != constrained_count
        ):
            raise ValueError(
                f"`nadir_points` length ({len(self.nadir_points)}) must match "
                f"constrained objectives ({constrained_count})."
            )

        if (
            self.sample_points is not None
            and len(self.sample_points) != 1
            and len(self.sample_points) != constrained_count
        ):
            raise ValueError(
                f"`sample_points` length ({len(self.sample_points)}) must match "
                f"constrained objectives ({constrained_count})."
            )

    def get_points_per_objective(self, constrained_count: int) -> list[int]:
        """Return the sample point count for each constrained objective.

        For exact mode this returns an empty list (counts derived from range).
        For sampled mode, broadcasts a single-element list to all objectives.
        """
        if self.sample_points is None:
            return []
        if len(self.sample_points) == 1:
            return self.sample_points * constrained_count
        return list(self.sample_points)

"""Public solver entry point for PyAugmecon runs."""

from __future__ import annotations

import csv
import threading
import time
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from multiprocessing import get_context
from multiprocessing.shared_memory import SharedMemory
from pathlib import Path
from queue import SimpleQueue
from types import SimpleNamespace
from typing import Any

import cloudpickle
import pyomo.environ as pyo
from pymoo.config import Config

from pyaugmecon import __version__
from pyaugmecon.config import PyAugmeconConfig
from pyaugmecon.logs import configure_loguru, log_run_header, log_run_summary
from pyaugmecon.results import PyAugmeconResult, WorkerChunk
from pyaugmecon.solver.model import Model, check_user_model
from pyaugmecon.solver.process import ProcessHandler
from pyaugmecon.solver.queue import QueueHandler
from pyaugmecon.solver.worker import (
    SkipBuffers,
    WorkerSpec,
    solver_worker_main,
)


def _write_csv(path: Path, rows: Sequence[Sequence[object]]) -> None:
    """Write row-oriented data to a CSV file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        csv.writer(handle).writerows(rows)


class PyAugmecon:
    """High-level AUGMECON solver wrapper around a Pyomo model."""

    def __init__(
        self,
        model: pyo.ConcreteModel,
        config: PyAugmeconConfig | Mapping[str, Any],
        *,
        log_sink: Any | None = None,
    ) -> None:
        config = PyAugmeconConfig.model_validate(config)
        if log_sink is not None and not callable(getattr(log_sink, "info", None)):
            raise TypeError("`log_sink` must expose an `.info(message)` method.")

        if not hasattr(model, "obj_list"):
            raise ValueError(
                "Model must define an `obj_list` with at least two objectives."
            )
        config.validate_against_model(len(model.obj_list))  # ty: ignore[invalid-argument-type]

        self.config = config

        # Microsecond-precise run id so concurrent runs don't clobber each other.
        self.run_id = datetime.now(tz=UTC).strftime("%Y%m%d-%H%M%S-%f")
        self.artifact_name = config.artifact_name or f"{config.name}_{self.run_id}"
        artifact_root = Path.cwd() / config.artifact_folder
        self.artifact_root = str(artifact_root)
        self.artifact_dir = str(artifact_root / self.artifact_name)
        self.logfile = str(artifact_root / f"{self.artifact_name}.log")

        configure_loguru(self.logfile, config.log_to_console, log_sink)
        check_user_model(model)
        log_run_header(config, __version__, self.artifact_dir)

        self.model = Model(model, config)

        # Suppress pymoo's "not compiled" warning. We don't ship the
        # cythonized variant and the pure-Python path is fine for our use.
        Config.warnings["not_compiled"] = False

        self.result: PyAugmeconResult | None = None
        self._model_blob_shm: SharedMemory | None = None

    def _require_result(self) -> PyAugmeconResult:
        if self.result is None:
            raise RuntimeError("Call `solve()` before requesting results.")
        return self.result

    def _find_solutions(self) -> None:
        """Dispatch grid work to workers and collect raw solution payloads.

        Two paths:

        * `workers > 1`: pickle the model into a SharedMemory block, spawn
          worker processes, and drain results from a multiprocessing queue.
          Cleanup of the shared block is unconditional via `finally`.
        * `workers == 1`: skip multiprocessing entirely and run the worker
          loop in the main process against the live Pyomo model. Avoids the
          spawn + pickle + IPC overhead that otherwise dominates small
          problems.
        """
        self.model.progress.set_message("Solving")
        if self.config.workers <= 1:
            self._find_solutions_inprocess()
            return
        self._find_solutions_multiprocess()

    def _find_solutions_inprocess(self) -> None:
        """Run the solver loop in the main process for `workers == 1`.

        Avoids `spawn + cloudpickle + IPC` overhead, which dominates small
        problems. Uses stdlib `SimpleQueue` and `threading.Event` because the
        worker only calls `.get/.put` and `.is_set/.set`; both are duck-
        compatible with their multiprocessing counterparts.
        """
        self.queues = QueueHandler(
            range(self.model.grid_point_count),
            self.model.grid_point_count,
            self.config,
            tuple(self.model.grid_sizes_inner),
        )
        self.queues.split_work(SimpleNamespace(Queue=SimpleQueue))

        spec = WorkerSpec.from_model(self.model, shm_name="", blob_size=0)
        skip_buffers = SkipBuffers(
            flag_buffer=None,
            flag_shape=tuple(self.model.grid_sizes_inner),
            flag_is_shared=False,
            outer_skip_buffer=None,
            outer_skip_shape=None,
        )

        result_q = self.queues.result_q
        error_q = self.queues.error_q
        assert result_q is not None
        assert error_q is not None

        solver_worker_main(
            worker_id=0,
            config=self.config,
            logfile=self.logfile,
            spec=spec,
            job_queue=self.queues.job_q_for_worker(0),
            result_q=result_q,
            error_q=error_q,
            stop_event=threading.Event(),
            visited_counter=self.model.progress.counter,
            solved_counter=self.model.models_solved,
            infeasible_counter=self.model.infeasibilities,
            skip_buffers=skip_buffers,
            live_model=self.model.model,
        )

        self._worker_chunks: list[WorkerChunk] = list(self.queues.get_result())
        self.model.progress.refresh()
        errors = self.queues.get_error()
        if errors:
            message = "\n\n".join(
                f"In-process worker failed (backend={e['backend']}):\n{e['traceback']}"
                for e in errors
            )
            raise RuntimeError(message)

    def _find_solutions_multiprocess(self) -> None:
        """Spawn worker processes and gather their solutions."""
        model_blob = cloudpickle.dumps(self.model.model)

        self._model_blob_shm = SharedMemory(create=True, size=len(model_blob))
        shm_buf = self._model_blob_shm.buf
        assert shm_buf is not None
        shm_buf[: len(model_blob)] = model_blob

        ctx = get_context("spawn")
        self.queues = QueueHandler(
            range(self.model.grid_point_count),
            self.model.grid_point_count,
            self.config,
            tuple(self.model.grid_sizes_inner),
        )
        self.queues.split_work(ctx)

        self.procs = ProcessHandler(
            self.config,
            self.model,
            self.queues,
            WorkerSpec.from_model(
                self.model, self._model_blob_shm.name, len(model_blob)
            ),
            logfile=self.logfile,
        )

        self._worker_chunks = []
        try:
            self.procs.start()
            # Cooperative wait loop: drain results periodically so the result
            # queue doesn't grow unbounded while workers are still running.
            while not self.procs.join():
                self.model.progress.refresh()
                self._worker_chunks.extend(self.queues.get_result())

            # Final drain after all workers exited cleanly.
            self.model.progress.refresh()
            self._worker_chunks.extend(self.queues.get_result())
        finally:
            self.procs.close()
            self._model_blob_shm.close()
            self._model_blob_shm.unlink()
            self._model_blob_shm = None

    def _build_result(self, runtime_seconds: float) -> None:
        visited = max(
            0, self.model.progress.counter.value() - self.model.setup_solve_count
        )
        self.result = PyAugmeconResult.from_worker_chunks(
            self._worker_chunks,
            sign=tuple(self.model.obj_goal),
            payoff_table=self.model.payoff,
            runtime_seconds=runtime_seconds,
            models_solved=self.model.models_solved.value(),
            models_infeasible=self.model.infeasibilities.value(),
            visited_points=visited,
            grid_point_count=self.model.grid_point_count,
            decision_variables_stored=self.config.store_decision_variables,
            round_decimals=self.config.round_decimals,
        )

    def _output_tables(self) -> None:
        """Write the epsilon grid, payoff table, and solutions to CSV files."""
        result = self._require_result()
        out = Path(self.artifact_dir)
        _write_csv(out / "epsilon_grid.csv", self.model.epsilon_grid.tolist())
        _write_csv(out / "payoff_table.csv", result.payoff_table.tolist())
        _write_csv(out / "solutions.csv", [list(point) for point in result.points])

    def solve(self) -> PyAugmeconResult:
        """Solve the multi-objective problem and return a structured result.

        The orchestration follows AUGMECON's standard staged pipeline:
          1. Deactivate all objectives so we control activation explicitly.
          2. Normalize every objective to maximization (`min_to_max`).
          3. Build the lexicographic payoff table (gives the per-objective
             ranges used by the augmented penalty term).
          4. Resolve nadir points and build the per-objective epsilon grids.
          5. Convert to augmented epsilon-constraint form (adds slack vars,
             epsilon param, augmented primary objective).
          6. Dispatch the grid to workers, then build the structured result.
          7. Optionally write CSV artifacts.

        The progress bar is closed in `finally` so a partial run still leaves
        the terminal in a sane state.
        """
        started_at = time.perf_counter()
        try:
            self.result = None
            self.model.deactivate_all_objectives()
            self.model.min_to_max()
            self.model.construct_payoff()
            self.model.find_obj_range()
            self.model.convert_prob()

            self._find_solutions()
            self._build_result(round(time.perf_counter() - started_at, 2))
            if self.config.write_csv:
                self._output_tables()
            log_run_summary(self._require_result())
            return self._require_result()
        finally:
            self.model.progress.close()

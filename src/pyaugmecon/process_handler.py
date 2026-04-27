"""Worker lifecycle orchestration for multiprocessing solve runs."""

from __future__ import annotations

import time
from math import prod
from multiprocessing import get_context
from multiprocessing.process import BaseProcess

import numpy as np
from loguru import logger as log

from pyaugmecon.config import PyAugmeconConfig
from pyaugmecon.model import Model
from pyaugmecon.queue_handler import QueueHandler
from pyaugmecon.solver_process import SkipBuffers, WorkerSpec, solver_worker_main


class ProcessHandler:
    """Manage worker startup, supervision, and teardown for one solve run."""

    def __init__(
        self,
        config: PyAugmeconConfig,
        model: Model,
        queues: QueueHandler,
        worker_spec: WorkerSpec,
        *,
        logfile: str,
    ):
        self.config = config
        self.model = model
        self.queues = queues
        self.worker_spec = worker_spec
        self.logfile = logfile

        self.ctx = get_context("spawn")
        self.stop_event = self.ctx.Event()
        self.procs: list[BaseProcess] = []
        self._skip_buffers: SkipBuffers | None = None
        self._started_at: float = 0.0

    def start(self) -> None:
        """Start worker processes."""
        self._started_at = time.perf_counter()
        self._skip_buffers = self._build_skip_buffers()

        self.procs = [
            self.ctx.Process(
                target=solver_worker_main,
                name=f"pyaugmecon-worker-{worker_id}",
                args=(
                    worker_id,
                    self.config,
                    self.logfile,
                    self.worker_spec,
                    self.queues.job_q_for_worker(worker_id),
                    self.queues.result_q,
                    self.queues.error_q,
                    self.stop_event,
                    self.model.progress.counter,
                    self.model.models_solved,
                    self.model.infeasibilities,
                    self._skip_buffers,
                ),
            )
            for worker_id in range(self.queues.worker_count)
        ]
        for proc in self.procs:
            proc.start()

    def join(self) -> bool:
        """Poll workers; return True once all have exited cleanly."""
        timeout = self.config.process_timeout
        if timeout is not None and time.perf_counter() - self._started_at > timeout:
            log.info("Timed out.")
            self.terminate_early()
            raise TimeoutError(f"Process timeout reached after {timeout} seconds.")

        for proc in self.procs:
            proc.join(timeout=0.1)

        # Workers report exceptions via error_q; check that before exit codes
        # so we surface the real traceback rather than just a nonzero status.
        errors = self.queues.get_error()
        if errors:
            self.terminate_early()
            errors.extend(self.queues.get_error())
            message = "\n\n".join(
                f"Worker {e['worker_id']} failed (backend={e['backend']}):\n{e['traceback']}"
                for e in errors
            )
            raise RuntimeError(message)

        if any(p.exitcode not in (None, 0) for p in self.procs):
            self.terminate_early()
            raise RuntimeError("At least one worker exited unexpectedly.")

        return all(p.exitcode is not None for p in self.procs)

    def terminate_early(self) -> None:
        """Stop all workers. Signal cooperatively, then SIGTERM stragglers."""
        self.stop_event.set()
        for proc in self.procs:
            proc.join(timeout=0.2)
            if proc.is_alive():
                proc.terminate()
                proc.join(timeout=0.2)

    def close(self) -> None:
        self._skip_buffers = None

    def _build_skip_buffers(self) -> SkipBuffers:
        """Allocate the AUGMECON-R skip tables for the active run.

        Two independent stores share one struct because they always travel
        together to workers:

        * Flag table (`config.flag`): per-cell skip counts. Shared mode uses a
          `lock=False` ctypes array; updates are advisory and monotone (workers
          only ratchet a cell upward via `np.maximum`), so a missed update
          under a race only causes a re-solve, not a wrong answer.
        * Outer-grid skip table (`work_distribution='outer_grid'`): per
          slower-changing-objective combination, the first inner index known
          to be infeasible. Same race-tolerance argument as flags.
        """
        flag_shape = tuple(self.model.grid_sizes_inner)
        flag_buffer = None
        flag_is_shared = False
        if self.config.flag and self.config.flag_policy == "shared":
            flag_buffer = self.ctx.Array("I", self.model.grid_point_count, lock=False)
            flag_is_shared = True

        outer_skip_buffer = None
        outer_skip_shape: tuple[int, ...] | None = None
        if (
            self.config.work_distribution == "outer_grid"
            and len(self.model.grid_sizes_inner) > 1
        ):
            outer_skip_shape = tuple(self.model.grid_sizes_inner[1:])
            outer_cells = int(prod(outer_skip_shape))
            if outer_cells > 0:
                outer_skip_buffer = self.ctx.Array("I", outer_cells, lock=False)
                view = np.ctypeslib.as_array(outer_skip_buffer).reshape(
                    outer_skip_shape, order="F"
                )
                view.fill(self.model.grid_sizes_inner[0])
            else:
                outer_skip_shape = None

        return SkipBuffers(
            flag_buffer=flag_buffer,
            flag_shape=flag_shape,
            flag_is_shared=flag_is_shared,
            outer_skip_buffer=outer_skip_buffer,
            outer_skip_shape=outer_skip_shape,
        )

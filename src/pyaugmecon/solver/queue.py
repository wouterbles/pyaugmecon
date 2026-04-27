"""Queue setup for worker jobs, results, and errors."""

from __future__ import annotations

import queue
from math import prod
from multiprocessing import Queue
from typing import TypedDict

from pyaugmecon.config import PyAugmeconConfig
from pyaugmecon.results import WorkerChunk


class WorkerError(TypedDict):
    worker_id: int
    backend: str
    traceback: str


class QueueHandler:
    """Split a flat grid range into worker jobs."""

    def __init__(
        self,
        work: range,
        work_size: int,
        config: PyAugmeconConfig,
        grid_sizes_inner: tuple[int, ...] | None = None,
    ):
        self.work = work
        self.work_size = int(work_size)
        self.config = config
        self.grid_sizes_inner = grid_sizes_inner

        self.worker_count = 0

        self.shared_job_q: Queue | None = None
        self.job_q_by_worker: list[Queue] | None = None
        self.result_q: Queue | None = None
        self.error_q: Queue | None = None

    @staticmethod
    def _partition_counts(total: int, buckets: int) -> list[int]:
        """Split `total` items into `buckets` near-equal sizes (bigger buckets first)."""
        base, remainder = divmod(total, buckets)
        return [base + (1 if i < remainder else 0) for i in range(buckets)]

    def _outer_alignment(self) -> tuple[int, int] | None:
        """Return `(outer_size, inner_block)` if the grid splits cleanly on the
        outer objective, else `None`.

        Splitting at outer-objective boundaries keeps related epsilon-level
        groups together on one worker, which makes pruning checks more likely
        to reuse information from previous solves in the same group.
        """
        if not self.grid_sizes_inner:
            return None
        outer_size = int(self.grid_sizes_inner[-1])
        if outer_size <= 0 or self.work_size % outer_size != 0:
            return None
        return outer_size, self.work_size // outer_size

    def _dynamic_chunk_size(self) -> int:
        """Pick a small job size for dynamic scheduling.

        Aim for ~8 jobs per worker so fast workers can pick up additional
        chunks. On large grids (>=128 points per worker), floor the chunk at
        8 to keep per-read queue overhead from dominating the solve time.
        Cap at 8192 as a safety net for pathological grid sizes.
        """
        if self.work_size <= self.worker_count:
            return 1
        chunk = self.work_size // (self.worker_count * 8)
        if self.work_size >= self.worker_count * 128:
            chunk = max(8, chunk)
        return max(1, min(8192, chunk))

    def _build_dynamic_ranges(self) -> list[range]:
        """Ranges for the shared dynamic queue.

        Prefers outer-aligned chunks when there are >=4 outer levels per
        worker so each chunk still covers related epsilon-level groups.
        """
        chunk = self._dynamic_chunk_size()
        alignment = self._outer_alignment()
        if alignment is not None and alignment[0] >= self.worker_count * 4:
            outer_size, inner_block = alignment
            outer_chunk = min(max(1, chunk // inner_block), outer_size)
            return [
                range(s * inner_block, min(s + outer_chunk, outer_size) * inner_block)
                for s in range(0, outer_size, outer_chunk)
            ]

        start, stop = int(self.work.start), int(self.work.stop)
        return [range(c, min(c + chunk, stop)) for c in range(start, stop, chunk)]

    def _build_outer_grid_ranges(self) -> list[range]:
        """Ranges aligned on outer-objective combinations.

        Each range covers one or more full outer-objective levels so the
        shared outer-grid skip table inside workers stays effective.
        """
        if not self.grid_sizes_inner:
            raise ValueError(
                "`work_distribution='outer_grid'` requires grid sizes. "
                "PyAugmecon provides them after it builds the epsilon grid."
            )
        if (
            not isinstance(self.work, range)
            or self.work.step != 1
            or int(self.work.start) != 0
            or int(self.work.stop) != self.work_size
        ):
            raise ValueError(
                "`work_distribution='outer_grid'` expects one continuous range "
                "from 0 to `work_size` so every grid point is assigned once."
            )

        # Use more work blocks than workers so fast workers can keep taking
        # another block, while each block still covers related objective levels.
        target_blocks = self.worker_count * 4

        # Pick the smallest outer-objective depth (counted from the slowest-
        # changing dimension) that produces at least target_blocks units.
        outer_sizes = tuple(reversed(self.grid_sizes_inner))
        depth = len(outer_sizes)
        block_unit_count = 1
        for d, size in enumerate(outer_sizes, start=1):
            block_unit_count *= int(size)
            if block_unit_count >= target_blocks:
                depth = d
                break

        block_unit_size = (
            int(prod(self.grid_sizes_inner[: len(self.grid_sizes_inner) - depth])) or 1
        )
        if block_unit_count <= 0:
            raise ValueError(
                "`work_distribution='outer_grid'` produced no work blocks; "
                "check the epsilon grid dimensions."
            )

        units_per_task = self._partition_counts(
            block_unit_count, min(block_unit_count, target_blocks)
        )
        ranges: list[range] = []
        cursor = 0
        for unit_count in units_per_task:
            if unit_count <= 0:
                continue
            start = cursor * block_unit_size
            cursor += unit_count
            ranges.append(range(start, cursor * block_unit_size))
        return ranges

    def _build_fixed_ranges(self) -> list[range | None]:
        """One range per worker (or None for an idle worker), preferring
        outer-aligned splits.
        """
        if not self.grid_sizes_inner:
            raise ValueError(
                "`work_distribution='fixed'` requires grid sizes. "
                "PyAugmecon provides them after it builds the epsilon grid."
            )

        alignment = self._outer_alignment()
        if alignment is not None:
            outer_size, inner_block = alignment
            counts = self._partition_counts(outer_size, self.worker_count)
            scale = inner_block
        else:
            counts = self._partition_counts(self.work_size, self.worker_count)
            scale = 1

        ranges: list[range | None] = []
        cursor = 0
        for count in counts:
            if count > 0:
                ranges.append(range(cursor * scale, (cursor + count) * scale))
            else:
                ranges.append(None)
            cursor += count
        return ranges

    def split_work(self, ctx) -> None:
        """Create shared queues and enqueue work for the selected distribution."""
        if self.work_size <= 0:
            raise ValueError("No work to split. Check objective count and exact grid.")

        self.worker_count = min(self.config.workers, self.work_size)
        self.result_q = ctx.Queue()
        self.error_q = ctx.Queue()

        if self.config.work_distribution == "fixed":
            self.job_q_by_worker = [ctx.Queue() for _ in range(self.worker_count)]
            for worker_q, r in zip(
                self.job_q_by_worker, self._build_fixed_ranges(), strict=False
            ):
                if r is not None:
                    worker_q.put(r)
                worker_q.put(None)
            return

        if self.config.work_distribution == "outer_grid":
            ranges = self._build_outer_grid_ranges()
        else:
            ranges = self._build_dynamic_ranges()

        self.shared_job_q = ctx.Queue()
        for r in ranges:
            self.shared_job_q.put(r)
        for _ in range(self.worker_count):
            self.shared_job_q.put(None)

    def job_q_for_worker(self, worker_id: int) -> Queue:
        if self.shared_job_q is not None:
            return self.shared_job_q
        if self.job_q_by_worker is None:
            raise RuntimeError("Job queue(s) are not initialized.")
        return self.job_q_by_worker[worker_id]

    @staticmethod
    def _drain_queue(q: Queue | None) -> list:
        """Drain all currently available items from a multiprocessing queue.

        The first read waits 10ms because `Queue.put` enqueues into an
        in-process buffer that a background feeder thread flushes to the
        underlying pipe; an immediate `get_nowait` can race that flush and
        miss items. Subsequent reads in the same drain are immediate.
        """
        if q is None:
            return []
        items: list = []
        while True:
            try:
                items.append(q.get(timeout=0.01) if not items else q.get_nowait())
            except queue.Empty:
                return items

    def get_result(self) -> list[WorkerChunk]:
        """Drain all currently available results from the shared result queue."""
        return self._drain_queue(self.result_q)

    def get_error(self) -> list[WorkerError]:
        """Drain all currently available worker error payloads."""
        return self._drain_queue(self.error_q)

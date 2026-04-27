from __future__ import annotations

import ctypes
import threading
import traceback
from dataclasses import dataclass
from multiprocessing import Queue
from multiprocessing.shared_memory import SharedMemory
from multiprocessing.synchronize import Event as MpEvent
from typing import TYPE_CHECKING, Any

import cloudpickle
import numpy as np
import pyomo.environ as pyo
from loguru import logger as log
from pyomo.core.base import Var
from pyomo.environ import value as pyo_value

from pyaugmecon.config import PyAugmeconConfig
from pyaugmecon.helper import Counter
from pyaugmecon.logs import configure_loguru
from pyaugmecon.results import Solution, WorkerChunk
from pyaugmecon.solver_adapter import (
    SolveOutcome,
    release_solver,
    select_solver,
    solve_once,
)

# `solver_worker_main` accepts either `multiprocessing.Event` (real workers)
# or `threading.Event` (workers=1 in-process fast path); both expose the same
# `is_set/set` API.
type StopEvent = MpEvent | threading.Event

if TYPE_CHECKING:
    from pyaugmecon.model import Model

# `multiprocessing.get_context('spawn').Array('I', n, lock=False)` returns a
# dynamically-named subclass of `ctypes.Array[c_uint]`. Alias for cross-process
# buffer typing.
type SharedUIntArray = "ctypes.Array[ctypes.c_uint]"

# Mirror of the same constant in `model.py`. Kept local so the worker hot loop
# avoids a cross-module attribute lookup per grid point.
_INFEASIBLE_OUTCOMES = frozenset(
    {SolveOutcome.INFEASIBLE, SolveOutcome.INFEASIBLE_OR_UNBOUNDED}
)


@dataclass(frozen=True, slots=True)
class WorkerSpec:
    """Immutable payload shared with each worker process."""

    model_shm_name: str
    model_shm_size: int
    grid_sizes_inner: tuple[int, ...]
    constrained_order_inner: tuple[int, ...]
    epsilon_levels_inner: tuple[tuple[float, ...], ...]
    level_steps_inner: tuple[float, ...]
    obj_range_by_obj: dict[int, float]

    @classmethod
    def from_model(cls, model: Model, shm_name: str, blob_size: int) -> WorkerSpec:
        """Snapshot the per-worker view of a configured `Model`."""
        order_inner = tuple(model.constrained_order_inner)
        return cls(
            model_shm_name=shm_name,
            model_shm_size=blob_size,
            grid_sizes_inner=tuple(model.grid_sizes_inner),
            constrained_order_inner=order_inner,
            epsilon_levels_inner=tuple(
                tuple(map(float, model.epsilon_levels_by_obj[obj_idx]))
                for obj_idx in order_inner
            ),
            level_steps_inner=tuple(
                float(model.level_step_by_obj[obj_idx]) for obj_idx in order_inner
            ),
            obj_range_by_obj={
                int(obj_idx): float(range_val)
                for obj_idx, range_val in model.obj_range_by_obj.items()
            },
        )


@dataclass(frozen=True, slots=True)
class SkipBuffers:
    """Shared-memory buffers and shapes for the AUGMECON-R skip tables.

    Workers turn these into numpy views via `SkipContext.from_buffers`.
    `flag_is_shared` distinguishes a shared ctypes buffer from a per-worker
    private array when `flag_buffer is None` (i.e. `config.flag` is on but
    `flag_policy != 'shared'`).
    """

    flag_buffer: SharedUIntArray | None
    flag_shape: tuple[int, ...]
    flag_is_shared: bool
    outer_skip_buffer: SharedUIntArray | None
    outer_skip_shape: tuple[int, ...] | None


@dataclass(frozen=True, slots=True)
class SkipContext:
    """Resolved per-worker views over the AUGMECON-R skip tables.

    Both views are always `np.ndarray`; disabled features use a 0-d sentinel
    so the per-iteration enabled-check stays a single bool.
    """

    flag_view: np.ndarray
    outer_skip_view: np.ndarray
    uses_outer_skip: bool

    @classmethod
    def from_buffers(
        cls, config: PyAugmeconConfig, buffers: SkipBuffers
    ) -> SkipContext:
        # Flag view: shared ctypes buffer, private numpy array, or 0-d sentinel.
        # Concurrent monotone (max) writes on the shared buffer are safe under
        # AUGMECON-R: flags are advisory, so a missed write only causes a
        # re-solve, never a wrong answer.
        if buffers.flag_buffer is not None:
            flag_view = np.ctypeslib.as_array(buffers.flag_buffer).reshape(
                buffers.flag_shape, order="F"
            )
        elif config.flag and not buffers.flag_is_shared:
            flag_view = np.zeros(buffers.flag_shape, dtype=np.uint32, order="F")
        else:
            flag_view = np.zeros((), dtype=np.uint32)

        # Outer-skip view: shared ctypes buffer or 0-d sentinel.
        if buffers.outer_skip_buffer is not None and buffers.outer_skip_shape:
            outer_skip_view = np.ctypeslib.as_array(buffers.outer_skip_buffer).reshape(
                buffers.outer_skip_shape, order="F"
            )
            uses_outer_skip = True
        else:
            outer_skip_view = np.zeros((), dtype=np.uint32)
            uses_outer_skip = False

        return cls(
            flag_view=flag_view,
            outer_skip_view=outer_skip_view,
            uses_outer_skip=uses_outer_skip,
        )


def _load_worker_model(spec: WorkerSpec) -> pyo.ConcreteModel:
    """Load the pickled Pyomo model out of the shared-memory blob.

    Slicing the buffer view directly avoids one full memcpy per worker that
    `bytes(view)` would incur for large user models.
    """
    model_shm = SharedMemory(name=spec.model_shm_name)
    try:
        shm_view = model_shm.buf
        assert shm_view is not None
        return cloudpickle.loads(shm_view[: spec.model_shm_size])
    finally:
        model_shm.close()


def _decode_point(linear_idx: int, grid_sizes: tuple[int, ...]) -> tuple[int, ...]:
    """Decode a flat job id into per-dimension coordinates.

    Dimension 0 is the innermost loop. The per-grid-point cost is dwarfed by
    the solve itself.
    """
    coords = [0] * len(grid_sizes)
    remainder = linear_idx
    for d, size in enumerate(grid_sizes):
        remainder, coords[d] = divmod(remainder, size)
    return tuple(coords)


def solver_worker_main(
    worker_id: int,
    config: PyAugmeconConfig,
    logfile: str,
    spec: WorkerSpec,
    job_queue: Queue,
    result_q: Queue,
    error_q: Queue,
    stop_event: StopEvent,
    visited_counter: Counter,
    solved_counter: Counter,
    infeasible_counter: Counter,
    skip_buffers: SkipBuffers,
    *,
    live_model: pyo.ConcreteModel | None = None,
) -> None:
    """Worker loop: read grid ranges, solve useful points, skip known dead space.

    `live_model`: when set, skip the SharedMemory unpickle and use the model
    directly. Used by the in-process workers=1 fast path.
    """
    if config.process_logging:
        configure_loguru(logfile, config.log_to_console)

    backend = "unresolved"
    solver: Any | None = None

    try:
        model = live_model if live_model is not None else _load_worker_model(spec)
        solver, selection = select_solver(config)
        backend = selection.resolved_backend

        skip = SkipContext.from_buffers(config, skip_buffers)

        # Resolve Pyomo components once; this loop can run thousands of times.
        obj_list = model.obj_list
        n_obj = len(obj_list)  # ty: ignore[invalid-argument-type]
        objective_exprs: tuple[Any, ...] = tuple(
            obj_list[obj_idx + 1].expr  # ty: ignore[not-subscriptable]
            for obj_idx in range(n_obj)
        )
        model_vars: tuple[Var, ...] = (
            tuple(model.component_map(ctype=Var, active=True).values())
            if config.store_decision_variables
            else ()
        )

        epsilon_param = model._pyaugmecon_eps
        slack_var = model._pyaugmecon_slack

        for obj_idx in range(n_obj):
            obj_list[obj_idx + 1].deactivate()  # ty: ignore[not-subscriptable]
        obj_list[1].activate()  # ty: ignore[not-subscriptable]

        component_indices = tuple(
            obj_idx + 1 for obj_idx in spec.constrained_order_inner
        )
        eps_data = tuple(epsilon_param[c] for c in component_indices)  # ty: ignore[not-subscriptable]
        slack_data = tuple(slack_var[c] for c in component_indices)  # ty: ignore[not-subscriptable]

        grid_sizes = spec.grid_sizes_inner
        epsilon_levels = spec.epsilon_levels_inner
        level_steps = spec.level_steps_inner
        last_index = tuple(size - 1 for size in grid_sizes)
        inner_dim_size = grid_sizes[0]
        last_inner = last_index[0]
        dim_count = len(grid_sizes)
        penalty_coeffs = tuple(
            (10 ** (-dim)) / spec.obj_range_by_obj[objective_idx]
            for dim, objective_idx in enumerate(spec.constrained_order_inner)
        )

        flag_enabled = config.flag
        bypass_enabled = config.bypass
        early_exit_enabled = config.early_exit
        store_vars = config.store_decision_variables
        warmstart = config.solve_warmstart
        penalty_weight = config.penalty_weight
        objective_tol = config.objective_tolerance

        flag_view = skip.flag_view
        outer_skip_view = skip.outer_skip_view
        uses_outer_skip = skip.uses_outer_skip
        outer_skip_shape = outer_skip_view.shape

        while not stop_event.is_set():
            work = job_queue.get()
            if work is None:
                break

            visited = 0
            solved = 0
            infeasible = 0
            results: WorkerChunk = []
            current_idx = int(work.start)
            linear_stop = int(work.stop)

            while current_idx < linear_stop:
                if stop_event.is_set():
                    break

                point = _decode_point(current_idx, grid_sizes)
                inner_index = point[0]
                outer_point = point[1:]
                visited += 1

                # Shared outer-grid skip: if another worker proved this
                # outer-objective combination infeasible from this inner
                # index onward, jump to the next combination.
                if uses_outer_skip:
                    first_blocked_inner = int(outer_skip_view[outer_point])
                    if inner_index >= first_blocked_inner:
                        jump = inner_dim_size - inner_index
                        visited += jump - 1
                        current_idx += jump
                        continue

                # Cell flags store how many inner points can be skipped here.
                if flag_enabled:
                    skip_count = int(flag_view[point])
                    if skip_count:
                        jump = min(skip_count, last_inner - inner_index + 1)
                        visited += jump - 1
                        current_idx += jump
                        continue

                for dim in range(dim_count):
                    eps_data[dim].value = epsilon_levels[dim][point[dim]]

                _, solve_result = solve_once(
                    model, solver, backend, warmstart=warmstart
                )
                solved += 1
                outcome = solve_result.outcome

                if outcome in _INFEASIBLE_OUTCOMES:
                    infeasible += 1
                    # Mark this and higher outer combinations as infeasible
                    # from the current inner index onward.
                    if uses_outer_skip and inner_index < int(
                        outer_skip_view[outer_point]
                    ):
                        slices = tuple(
                            slice(outer_point[d], outer_skip_shape[d])
                            for d in range(len(outer_skip_shape))
                        )
                        target = outer_skip_view[slices]
                        np.minimum(target, np.uint32(inner_index), out=target)

                    if early_exit_enabled:
                        # Outer-skip already covers this case; only write cell
                        # flags when outer-skip is off.
                        if flag_enabled and not uses_outer_skip:
                            slices_flag = (
                                slice(inner_index, inner_index + 1),
                                *tuple(
                                    slice(point[d], grid_sizes[d])
                                    for d in range(1, dim_count)
                                ),
                            )
                            flag_target = flag_view[slices_flag]
                            np.maximum(
                                flag_target,
                                last_inner - inner_index + 1,
                                out=flag_target,
                            )
                        jump = last_inner - inner_index
                        visited += jump
                        current_idx += 1 + jump
                    else:
                        current_idx += 1
                    continue

                if outcome != SolveOutcome.OPTIMAL:
                    raise RuntimeError(
                        "Worker encountered unsupported solver termination "
                        f"at grid point {point}. outcome={outcome}, "
                        f"status={solve_result.pyomo_status}, "
                        f"termination={solve_result.pyomo_termination}, "
                        f"backend={backend}."
                    )

                slack_values = [float(s.value or 0.0) for s in slack_data]

                bypass_jump = 0
                if bypass_enabled:
                    slack_steps = [
                        max(0, int((slack_values[dim] / level_steps[dim]) + 1e-9))
                        for dim in range(dim_count)
                    ]
                    bypass_jump = min(slack_steps[0], last_inner - inner_index)
                    if flag_enabled:
                        # Positive slack means the same solution satisfies
                        # nearby harder epsilon levels; flag those points so
                        # later iterations skip the solve.
                        slices_flag = (
                            slice(inner_index, inner_index + 1),
                            *tuple(
                                slice(
                                    point[d],
                                    min(last_index[d], point[d] + slack_steps[d]) + 1,
                                )
                                for d in range(1, dim_count)
                            ),
                        )
                        flag_target = flag_view[slices_flag]
                        np.maximum(flag_target, slack_steps[0] + 1, out=flag_target)

                # Report plain objectives; subtract the slack penalty from
                # f_0 (it's only an AUGMECON tie-breaker).
                penalty = sum(
                    coeff * slack
                    for coeff, slack in zip(penalty_coeffs, slack_values, strict=False)
                )
                objective_values: list[float] = [
                    float(pyo_value(expr)) for expr in objective_exprs
                ]
                objective_values[0] -= penalty_weight * penalty

                # Snap near-integer noise before keys go through the queue.
                for i, val in enumerate(objective_values):
                    rounded = round(val)
                    if abs(val - rounded) <= objective_tol:
                        objective_values[i] = float(rounded)

                results.append(
                    Solution(
                        point=tuple(objective_values),
                        variables={
                            var.name: {
                                k: float(v) for k, v in var.extract_values().items()
                            }
                            for var in model_vars
                        }
                        if store_vars
                        else None,
                    )
                )

                visited += bypass_jump
                current_idx += 1 + bypass_jump

            visited_counter.add(visited)
            solved_counter.add(solved)
            infeasible_counter.add(infeasible)
            if results:
                result_q.put(results)

        if config.process_logging:
            log.info(f"Process {worker_id} finished")
    except Exception:
        # BaseException subclasses (KeyboardInterrupt, SystemExit, MemoryError)
        # are intentionally not caught; they should crash the worker so the OS
        # surfaces the real cause.
        error_q.put(
            {
                "worker_id": worker_id,
                "backend": backend,
                "traceback": traceback.format_exc(),
            }
        )
        stop_event.set()
        raise
    finally:
        if solver is not None:
            release_solver(solver, backend)

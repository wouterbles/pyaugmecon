from __future__ import annotations

from multiprocessing import get_context
from unittest.mock import MagicMock

import pytest

from pyaugmecon import PyAugmecon
from pyaugmecon.solver.model import Model
from pyaugmecon.solver.process import ProcessHandler
from pyaugmecon.solver.queue import QueueHandler
from pyaugmecon.solver.worker import SkipBuffers, WorkerSpec
from tests.support.factories import make_config
from tests.support.models import three_objective_model, two_objective_model

_CTX = get_context("spawn")


def _setup_queues(model: Model, config) -> QueueHandler:
    queues = QueueHandler(
        work=range(model.grid_point_count),
        work_size=model.grid_point_count,
        config=config,
        grid_sizes_inner=tuple(model.grid_sizes_inner) or None,
    )
    queues.worker_count = config.workers
    queues.result_q = _CTX.Queue()
    queues.error_q = _CTX.Queue()
    return queues


@pytest.fixture
def handler():
    cfg = make_config("proc_test")
    py = PyAugmecon(two_objective_model(), cfg)
    queues = _setup_queues(py.model, cfg)
    return ProcessHandler(
        config=cfg,
        model=py.model,
        queues=queues,
        worker_spec=MagicMock(spec=WorkerSpec),
        logfile="",
    )


class TestTerminateEarly:
    def test_sets_stop_event(self, handler):
        proc = MagicMock()
        proc.is_alive.return_value = False
        handler.procs = [proc]

        handler.terminate_early()

        assert handler.stop_event.is_set()
        proc.join.assert_called()

    def test_kills_stragglers(self, handler):
        alive = MagicMock()
        alive.is_alive.return_value = True
        handler.procs = [alive]

        handler.terminate_early()

        alive.terminate.assert_called_once()


class TestJoin:
    def test_raises_timeout(self, handler, monkeypatch):
        handler.config = make_config("timeout_test", process_timeout=0.001)
        handler._started_at = 0.0
        monkeypatch.setattr(
            "pyaugmecon.solver.process.time.perf_counter", lambda: 999.0
        )

        with pytest.raises(TimeoutError, match="timeout"):
            handler.join()

    def test_collects_worker_errors(self, handler):
        error_q = handler.ctx.Queue()
        error_q.put({"worker_id": 0, "backend": "highs", "traceback": "boom"})
        handler.queues.error_q = error_q

        handler.procs = [MagicMock(exitcode=0)]

        with pytest.raises(RuntimeError, match="Worker 0 failed"):
            handler.join()

    def test_raises_on_nonzero_exit_code(self, handler):
        handler.queues.error_q = handler.ctx.Queue()
        handler.procs = [MagicMock(exitcode=1, is_alive=MagicMock(return_value=False))]

        with pytest.raises(RuntimeError, match="exited unexpectedly"):
            handler.join()


class TestClose:
    def test_clears_skip_buffers(self, handler):
        handler._skip_buffers = MagicMock(spec=SkipBuffers)

        handler.close()

        assert handler._skip_buffers is None


class TestBuildSkipBuffers:
    def test_shared_flag(self, handler):
        handler.config = make_config("flag_test", flag=True, flag_policy="shared")

        bufs = handler._build_skip_buffers()

        assert bufs.flag_buffer is not None
        assert bufs.flag_is_shared is True

    def test_no_flag_buffers_by_default(self, handler):
        bufs = handler._build_skip_buffers()

        assert bufs.flag_buffer is None
        assert bufs.flag_is_shared is False

    def test_outer_grid_three_objectives(self, handler):
        cfg = make_config(
            "skip_outer_test", work_distribution="outer_grid", flag_policy="local"
        )
        py = PyAugmecon(three_objective_model(), cfg)
        model = py.model
        model.deactivate_all_objectives()
        model.min_to_max()
        model.construct_payoff()
        model.find_obj_range()

        handler.config = cfg
        handler.model = model

        bufs = handler._build_skip_buffers()

        assert bufs.outer_skip_buffer is not None
        assert bufs.outer_skip_shape is not None
        assert len(bufs.outer_skip_shape) >= 1

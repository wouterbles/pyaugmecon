from multiprocessing import get_context

import pytest

from pyaugmecon.solver.queue import QueueHandler
from tests.support.factories import make_config


def test_split_work_requires_non_empty_work():
    queues = QueueHandler(range(0), 0, make_config("queue_test", workers=2))

    with pytest.raises(ValueError, match="No work"):
        queues.split_work(get_context("spawn"))


def test_split_work_initializes_shared_queues():
    work = range(10)
    queues = QueueHandler(
        range(len(work)),
        len(work),
        make_config("queue_test", workers=3, work_distribution="dynamic"),
    )

    queues.split_work(get_context("spawn"))

    assert queues.worker_count == 3
    assert queues.shared_job_q is not None
    assert queues.result_q is not None
    assert queues.error_q is not None


def test_get_result_drains_queue():
    work = range(1)
    queues = QueueHandler(
        range(len(work)), len(work), make_config("queue_test", workers=1)
    )
    queues.split_work(get_context("spawn"))

    assert queues.result_q is not None
    queues.result_q.put({"a": 1})
    queues.result_q.put({"b": 2})

    drained = queues.get_result()
    assert drained == [{"a": 1}, {"b": 2}]
    assert queues.get_result() == []


def test_split_work_fixed_initializes_worker_queues():
    work = range(12)
    config = make_config("queue_fixed", workers=3, work_distribution="fixed")
    queues = QueueHandler(range(len(work)), len(work), config, grid_sizes_inner=(3, 4))

    queues.split_work(get_context("spawn"))

    assert queues.worker_count == 3
    assert queues.shared_job_q is None
    assert queues.job_q_by_worker is not None
    assert len(queues.job_q_by_worker) == 3
    assert queues.result_q is not None
    assert queues.error_q is not None


def test_split_work_fixed_requires_grid_shape():
    work = range(12)
    config = make_config(
        "queue_fixed_missing_shape", workers=3, work_distribution="fixed"
    )
    queues = QueueHandler(range(len(work)), len(work), config)

    with pytest.raises(ValueError, match="fixed"):
        queues.split_work(get_context("spawn"))


def test_split_work_outer_grid_covers_full_grid_with_aligned_ranges():
    work = range(64)
    config = make_config("queue_outer_grid", workers=3, work_distribution="outer_grid")
    queues = QueueHandler(work, len(work), config, grid_sizes_inner=(4, 4, 4))

    queues.split_work(get_context("spawn"))

    assert queues.worker_count == 3
    assert queues.shared_job_q is not None
    assert queues.job_q_by_worker is None

    drained: list[object] = []
    sentinels = 0
    while sentinels < queues.worker_count:
        item = queues.shared_job_q.get(timeout=0.5)
        drained.append(item)
        if item is None:
            sentinels += 1

    blocks = [item for item in drained if isinstance(item, range)]
    assert blocks
    assert all(block.step == 1 for block in blocks)
    assert all((int(block.stop) - int(block.start)) % 4 == 0 for block in blocks)

    cursor = 0
    for block in blocks:
        assert int(block.start) == cursor
        cursor = int(block.stop)
    assert cursor == len(work)


def test_split_work_outer_grid_requires_linear_range_work():
    work = list(range(64))
    config = make_config(
        "queue_outer_grid_bad_work", workers=2, work_distribution="outer_grid"
    )
    queues = QueueHandler(work, len(work), config, grid_sizes_inner=(4, 4, 4))

    with pytest.raises(ValueError, match="one continuous range"):
        queues.split_work(get_context("spawn"))

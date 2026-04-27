"""Utility helpers shared across the PyAugmecon runtime."""

from multiprocessing import Value

from tqdm.auto import tqdm


class Counter:
    """A small process-safe integer counter backed by shared memory.

    Uses ``multiprocessing.Value`` which provides its own internal lock,
    so no separate ``Lock`` is needed.
    """

    def __init__(self, init_val: int = 0):
        self._val = Value("i", init_val)

    def add(self, amount: int = 1) -> None:
        """Atomically add `amount`. No-op when `amount == 0`."""
        if amount == 0:
            return
        with self._val.get_lock():
            self._val.value += amount

    def value(self) -> int:
        return self._val.value


class ProgressBar:
    """Wrap tqdm around a shared Counter for cross-process progress tracking.

    When `enabled=False` we still construct a tqdm with `disable=True`, which
    turns every method (`update`, `close`, `set_description_str`) into a no-op.
    This avoids `if self.bar is None` guards on the hot path.

    The bar lives only in the main process. Workers receive the underlying
    `Counter` directly (see `ProcessHandler.start`), so this object never
    needs to be pickled across the process boundary.
    """

    def __init__(self, counter: Counter, total: int, *, enabled: bool = True):
        self.counter = counter
        self.total = total
        self.last_value = counter.value()
        self.bar: tqdm = tqdm(
            total=total,
            desc="Progress",
            dynamic_ncols=True,
            leave=False,
            unit="model",
            mininterval=0.3,
            smoothing=0.1,
            disable=not enabled,
            bar_format=(
                "{desc}: {percentage:3.0f}%|{bar}| "
                "{n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]"
            ),
        )

    def set_message(self, message: str) -> None:
        self.bar.set_description_str(message or "Progress", refresh=False)

    def set_total(self, total: int) -> None:
        self.total = total
        self.bar.total = total
        self.bar.refresh()

    def refresh(self) -> None:
        current = min(self.counter.value(), self.total)
        if current > self.last_value:
            self.bar.update(current - self.last_value)
            self.last_value = current

    def increment(self) -> None:
        self.counter.add(1)
        self.refresh()

    def close(self) -> None:
        self.refresh()
        self.bar.close()

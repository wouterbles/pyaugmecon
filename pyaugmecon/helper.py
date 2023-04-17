import time
from multiprocessing import Lock, Value


class Helper:
    """A class of helper functions."""

    @staticmethod
    def clear_line():
        """Clears the current line in the console."""
        print(" " * 80, end="\r", flush=True)

    @staticmethod
    def separator():
        """Returns a string of '=' characters for use as a separator."""
        return "=" * 30

    @staticmethod
    def keys_to_list(d: dict):
        """Converts the keys of a dictionary to a list."""
        return [list(key) for key in d.keys()]


class Counter:
    """A thread-safe counter class."""

    def __init__(self, init_val: int = 0):
        """Initializes the counter with an optional initial value."""
        self.val = Value("i", init_val)
        self.lock = Lock()

    def increment(self):
        """Increments the counter."""
        with self.lock:
            self.val.value += 1

    def value(self):
        """Returns the current value of the counter."""
        with self.lock:
            return self.val.value


class Timer:
    """A class for measuring elapsed time."""

    def __init__(self):
        """Starts the timer."""
        self.start_time = time.time()

    def get(self):
        """Returns the elapsed time since the timer was started."""
        return time.time() - self.start_time


class ProgressBar:
    """A class for displaying a progress bar."""

    def __init__(self, counter: Counter, total: int, init_message: str = ""):
        """Initializes the progress bar with a counter and a total number of iterations."""
        self.counter = counter
        self.total = total
        self.message = init_message
        self.bar = ""

    def set_message(self, message):
        """Sets the message displayed next to the progress bar."""
        self.message = message
        Helper.clear_line()
        self.print(True)

    def print(self, force):
        """Prints the progress bar."""
        bar_len = 40

        progress = self.counter.value() / float(self.total)
        filled_len = int(round(bar_len * progress, 1))
        percents = round(100.0 * progress, 1)
        bar = "=" * filled_len + "-" * (bar_len - filled_len)

        if self.bar != bar or force:
            self.bar = bar
            print(f"[{bar}] {percents}% ... ({self.message})", end="\r", flush=True)

    def increment(self):
        """Increments the counter and updates the progress bar."""
        self.counter.increment()
        self.print(False)

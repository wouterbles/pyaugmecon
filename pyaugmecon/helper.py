import time
from multiprocessing import Value, Lock


class Helper(object):
    def clear_line():
        print(" " * 80, end="\r", flush=True)

    def separator():
        return "=" * 30


class Counter(object):
    def __init__(self, init_val: int = 0):
        self.val = Value("i", init_val)
        self.lock = Lock()

    def increment(self):
        with self.lock:
            self.val.value += 1

    def value(self):
        with self.lock:
            return self.val.value


class Timer(object):
    def __init__(self):
        self.start_time = time.time()

    def get(self):
        return time.time() - self.start_time


class ProgressBar(object):
    def __init__(self, counter: Counter, total: int, init_message: str = ""):
        self.counter = counter
        self.total = total
        self.message = init_message
        self.bar = ""

    def set_message(self, message):
        self.message = message
        Helper.clear_line()
        self.print(True)

    def print(self, force):
        bar_len = 40

        progress = self.counter.value() / float(self.total)
        filled_len = int(round(bar_len * progress, 1))
        percents = round(100.0 * progress, 1)
        bar = "=" * filled_len + "-" * (bar_len - filled_len)

        if self.bar != bar or force:
            self.bar = bar
            print(f"[{bar}] {percents}% ... ({self.message})", end="\r", flush=True)

    def increment(self):
        self.counter.increment()
        self.print(False)

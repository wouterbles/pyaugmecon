import queue
from multiprocessing import Queue
from pyaugmecon.options import Options


class QueueHandler(object):
    def __init__(self, work, opts: Options):
        self.work = work
        self.opts = opts
        self.job_qs = [Queue() for _ in range(self.opts.cpu_count)]
        self.result_q = Queue()

    def get_longest_q(self):
        q_length = [q.qsize() for q in self.job_qs]
        if all(q == 0 for q in q_length):
            return None
        else:
            return q_length.index(max(q_length))

    def get_work(self, i):
        try:
            return self.job_qs[i].get_nowait()
        except queue.Empty:
            if (self.get_longest_q()):
                return self.get_work(self.get_longest_q())
            else:
                return None

    def put_result(self, result):
        self.result_q.put(result)

    def get_result(self, procs):
        return [self.result_q.get() for _ in procs]

    def split_work(self):
        # Divide grid points in blocks
        blocks = [self.work[i:i + self.opts.gp]
                  for i in range(0, len(self.work), self.opts.gp)]

        remainder = self.opts.gp % self.opts.cpu_count
        take = int((self.opts.gp - remainder) / self.opts.cpu_count)
        work_split = []

        start = -take
        for i in range(self.opts.cpu_count):
            start += take
            end = start + take + remainder
            for w in blocks[start:end]:
                self.job_qs[i].put(w)

            if i == 0:
                start += remainder
                remainder = 0

        for i, w in enumerate(work_split):
            self.job_qs[i].put(w)

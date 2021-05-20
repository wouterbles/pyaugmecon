import queue
import logging
import numpy as np
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
            if (self.opts.redivide_work and self.get_longest_q()):
                return self.get_work(self.get_longest_q())
            else:
                logging.info(f'{i} exit')
                return None

    def put_result(self, result):
        self.result_q.put(result)

    def get_result(self, procs):
        return [self.result_q.get() for _ in procs]

    def split_work(self):
        blocks = [self.work[i:i + self.opts.gp]
                  for i in range(0, len(self.work), self.opts.gp)]
        blocks = np.array_split(np.array(blocks), self.opts.cpu_count)

        for i, b in enumerate(blocks):
            for item in b:
                item = [tuple(x) for x in item.tolist()]
                self.job_qs[i].put_nowait(item)

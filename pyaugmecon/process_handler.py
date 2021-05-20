from multiprocessing import Process
from pyaugmecon.options import Options
from pyaugmecon.flag import Flag


class ProcessHandler(object):
    def __init__(self, opts: Options, func, m, q):
        self.opts = opts
        self.flag = Flag(m, self.opts)

        self.procs = [Process(
            target=func,
            args=(p, self.opts, m, q, self.flag))
            for p in range(self.opts.cpu_count)]

    def start(self):
        for p in self.procs:
            p.start()

    def join(self):
        for p in self.procs:
            p.join()

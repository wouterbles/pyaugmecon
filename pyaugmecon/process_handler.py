from multiprocessing import Process
from pyaugmecon.options import Options
from pyaugmecon.flag import Flag
from pyaugmecon.logs import Logs


class ProcessHandler(object):
    def __init__(self, opts: Options, func, m, q, logs: Logs):
        self.opts = opts
        self.logger = logs.logger
        self.flag = Flag(self.opts)

        self.procs = [Process(
            target=func,
            args=(p, self.opts, m, q, self.flag, logs))
            for p in range(self.opts.cpu_count)]

    def start(self):
        self.logger.info(f'Starting {self.opts.cpu_count} worker processes')

        for p in self.procs:
            p.start()

    def join(self):
        self.logger.info(f'Joining {self.opts.cpu_count} worker processes')

        for p in self.procs:
            p.join()

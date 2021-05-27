import logging
from multiprocessing import Process
from pyaugmecon.options import Options
from pyaugmecon.flag import Flag


class ProcessHandler(object):
    def __init__(self, opts: Options, func, m, q):
        self.opts = opts
        self.logger = logging.getLogger(opts.log_name)
        self.logger.setLevel(logging.INFO)
        self.flag = Flag(self.opts)

        self.procs = [Process(
            target=func,
            args=(p, self.opts, m, q, self.flag))
            for p in range(self.opts.cpu_count)]

    def start(self):
        self.logger.info(f'Starting {self.opts.cpu_count} worker processes')

        for p in self.procs:
            p.start()

    def join(self):
        self.logger.info(f'Joining {self.opts.cpu_count} worker processes')

        for p in self.procs:
            p.join()

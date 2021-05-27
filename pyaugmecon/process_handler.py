import logging
from multiprocessing import Process
from pyaugmecon.options import Options
from pyaugmecon.flag import Flag
from pyaugmecon.queue_handler import QueueHandler
from pyaugmecon.model import Model


class ProcessHandler(object):
    def __init__(
            self,
            opts: Options,
            func,
            model: Model,
            queues: QueueHandler):

        self.opts = opts
        self.model = model
        self.queues = queues
        self.logger = logging.getLogger(opts.log_name)
        self.flag = Flag(self.opts)

        self.procs = [Process(
            target=func,
            args=(p, self.opts, self.model, self.queues, self.flag))
            for p in range(self.queues.proc_count)]

    def start(self):
        self.logger.info(f'Starting {self.queues.proc_count} worker process(es)')

        for p in self.procs:
            p.start()

    def join(self):
        self.logger.info(f'Joining {self.queues.proc_count} worker process(es)')

        for p in self.procs:
            p.join()

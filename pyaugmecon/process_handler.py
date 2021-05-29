import time
import logging
from threading import Thread
from pyaugmecon.flag import Flag
from pyaugmecon.model import Model
from pyaugmecon.helper import Timer
from pyaugmecon.options import Options
from pyaugmecon.queue_handler import QueueHandler
from pyaugmecon.solver_process import SolverProcess


class ProcessHandler(object):
    def __init__(self, opts: Options, model: Model, queues: QueueHandler):
        self.opts = opts
        self.model = model
        self.queues = queues
        self.logger = logging.getLogger(opts.log_name)
        self.flag = Flag(self.opts)

        if self.opts.process_timeout:
            self.timeout = Thread(target=self.check_timeout)

        self.procs = [
            SolverProcess(p_num, self.opts, self.model, self.queues, self.flag)
            for p_num in range(self.queues.proc_count)
        ]

    def start(self):
        self.runtime = Timer()
        self.logger.info(f"Starting {self.queues.proc_count} worker process(es)")

        for p in self.procs:
            p.start()

        if self.opts.process_timeout:
            self.timeout.start()

    def check_timeout(self):
        while self.runtime.get() <= self.opts.process_timeout:
            if not any(p.is_alive() for p in self.procs):
                break
            time.sleep(0.5)
        else:
            self.logger.info("Timed out, gracefully stopping all worker proces(es)")
            self.queues.empty_job_qs()

    def join(self):
        self.logger.info(f"Joining {self.queues.proc_count} worker process(es)")

        if self.opts.process_timeout:
            self.timeout.join()

        for p in self.procs:
            p.join()

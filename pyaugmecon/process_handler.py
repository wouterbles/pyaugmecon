import logging
import time
from threading import Thread

from pyaugmecon.flag import Flag
from pyaugmecon.helper import Timer
from pyaugmecon.model import Model
from pyaugmecon.options import Options
from pyaugmecon.queue_handler import QueueHandler
from pyaugmecon.solver_process import SolverProcess


class ProcessHandler:
    def __init__(self, opts: Options, model: Model, queues: QueueHandler):
        """
        Initialize the ProcessHandler object.

        Parameters
        ----------
        opts : Options
            An instance of Options class.
        model : Model
            An instance of Model class.
        queues : QueueHandler
            An instance of QueueHandler class.

        """
        self.opts = opts
        self.model = model
        self.queues = queues
        self.logger = logging.getLogger(opts.log_name)
        self.flag = Flag(opts)
        self.procs = []
        self.any_killed = False

        # Create a timer thread if process timeout is specified in the options
        if opts.process_timeout:
            self.timeout = Thread(target=self.check_timeout)

    def start(self):
        """
        Start the worker processes.
        """
        self.runtime = Timer()  # Start a timer to measure the runtime
        self.logger.info(f"Starting {self.queues.proc_count} worker process(es)")

        self.procs = [
            SolverProcess(p_num, self.opts, self.model, self.queues, self.flag)
            for p_num in range(self.queues.proc_count)
        ]  # Create a SolverProcess object for each process and store them in a list
        for p in self.procs:
            p.start()  # Start each process

        if self.opts.process_timeout:
            self.timeout.start()  # Start the timer thread if process timeout is specified

    def check_timeout(self):
        """
        Check whether the process timeout has been reached.
        """
        while self.runtime.get() <= self.opts.process_timeout:
            if not any(p.is_alive() for p in self.procs):  # Check if any process has exited
                break
            time.sleep(1)
        else:
            self.logger.info("Timed out.")
            self.terminate_early()

    def join(self):
        """
        Wait a bit for the worker processes to finish.
        Returns true when all of them have finished, false otherwise.
        """
        if self.opts.process_timeout:
            self.timeout.join()  # Wait for the timer thread to finish

        all_exited = True
        for p in self.procs:
            if not self.any_killed and p.exitcode != None and p.exitcode < 0:
                self.any_killed = True
                self.logger.warning("A worker was killed, computations are not guaranteed to be completed.")
                self.terminate_early()
            
            all_exited = p.join(1) == None and p.exitcode != None and all_exited

        return all_exited
        
    def terminate_early(self):
        self.logger.info("Gracefully stopping all workers.")
        self.queues.empty_job_qs()

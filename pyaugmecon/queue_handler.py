import logging
import queue
from multiprocessing import Queue

import numpy as np

from pyaugmecon.options import Options


class QueueHandler:
    def __init__(self, work, opts: Options):
        """
        Initialize a QueueHandler object.

        Parameters
        ----------
        work : Any
            The input work.
        opts : Options
            The options object.

        """
        self.work = work
        self.opts = opts
        self.logger = logging.getLogger(opts.log_name)
        self.result_q = Queue()
        self.proc_count = 0
        self.job_qs = []

    def get_longest_q(self):
        """
        Get the index of the job queue with the most items.

        Returns
        -------
        int or None
            The index of the job queue with the most items, or None if all queues are empty.

        """
        q_lengths = [q.qsize() for q in self.job_qs]
        if all(q == 0 for q in q_lengths):
            return None
        return np.argmax(q_lengths)

    def get_work(self, i):
        """
        Get work for the i-th process.

        Parameters
        ----------
        i : int
            Index of the process

        Returns
        -------
        list or None
            A list of items to process, or None if there is no more work.

        """
        try:
            return self.job_qs[i].get_nowait()  # Try to get the work for a given process without blocking
        except queue.Empty:  # If the queue is empty
            if self.opts.redivide_work and self.get_longest_q() is not None:
                # If the `redivide_work` flag is set to True and there is work available, redivide the work
                return self.get_work(self.get_longest_q())
            if self.opts.process_logging:
                # If process logging is enabled, log that the process exited
                self.logger.info(f"PID: {i} exited")
            self.result_q.put("STOP")  # Put "STOP" in the result queue to signal the end of processing
            return None

    def put_result(self, result):
        """
        Put a result in the result queue.

        Parameters
        ----------
        result : any
            The result to put in the queue.

        """
        self.result_q.put(result)  # Put a result in the result queue

    def get_result(self):
        """
        Get the results from the result queue.

        Returns
        -------
        list
            A list of results.

        """
        results = []  # List to hold the results
        for _ in range(self.proc_count):
            while True:
                result = self.result_q.get()
                if result == "STOP":
                    break  # If the result is "STOP", break out of the loop
                results.append(result)  # Otherwise, append the result to the list
        return results

    def empty_job_qs(self):
        """
        Empty the job queues.
        """
        for job_q in self.job_qs:
            while True:
                try:
                    job_q.get_nowait()
                except queue.Empty:
                    break  # Empty the job queues

    def split_work(self):
        """
        Split the input work into blocks and create a job queue for each process.
        """
        block_size = self.opts.gp  # Get the block size from the options
        blocks = [
            self.work[i : i + block_size] for i in range(0, len(self.work), block_size)
        ]  # Divide the work into blocks
        blocks = np.array_split(np.array(blocks), self.opts.cpu_count)  # Divide the blocks into sub-blocks
        blocks = [x for x in blocks if x.size > 0]  # Remove empty sub-blocks
        self.proc_count = len(blocks)  # Set the number of processes to be used
        self.job_qs = [Queue() for _ in range(self.proc_count)]  # Create a job queue for each process
        self.logger.info(f"Dividing grid over {self.proc_count} process(es)")  # Log the number of processes

        for i, block in enumerate(blocks):
            items = [tuple(item) for sublist in block.tolist() for item in sublist]  # Flatten the sub-blocks
            self.job_qs[i].put(items)  # Put the flattened items in the job queue for the process

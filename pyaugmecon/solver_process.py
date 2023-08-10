import logging
from multiprocessing import Process

from pyaugmecon.flag import Flag
from pyaugmecon.model import Model
from pyaugmecon.options import Options
from pyaugmecon.queue_handler import QueueHandler


class SolverProcess(Process):
    def __init__(self, p_num, opts: Options, model: Model, queues: QueueHandler, flag: Flag):
        """
        Initialize the SolverProcess object.

        Parameters
        ----------
        p_num : int
            The process number.
        opts : Options
            The options object.
        model : Model
            The model object.
        queues : QueueHandler
            The queue handler object.
        flag : Flag
            The flag object.

        """
        super().__init__()
        self.p_num = p_num
        self.opts = opts
        self.model = model
        self.queues = queues
        self.flag = flag
        self.logger = None

    def run(self):
        """
        Run the SolverProcess.

        This function runs the SolverProcess. It retrieves work items from the queue and processes them until there is
        no more work to do. For each work item, it sets up the model and solves it. If the `early_exit` flag is set to
        True and the model is infeasible, the function sets the flag and jumps to the next work item. If the `bypass`
        flag is set to True and the model is optimal, the function sets the flag and jumps to the next work item. If
        the model is optimal, the function calculates the solutions and puts them in the result queue.

        """
        jump = 0  # Initialize variable 'jump' to zero

        # Check if process logging is enabled
        if self.opts.process_logging:
            # If enabled, initialize logger with log name provided in 'self.opts.log_name'
            self.logger = logging.getLogger(self.opts.log_name)
            self.logger.setLevel(logging.INFO)  # Set log level to INFO

        # Load pickled model
        self.model.unpickle()

        # Run indefinitely until there's no more work
        while True:
            # Get work from queue for this process number
            work = self.queues.get_work(self.p_num)

            # If no more work, break the loop
            if not work:
                break

            # Process each job in the work list
            for c in work:
                log = f"Process: {self.p_num}, index: {c}, "  # Initialize logging string
                cp_end = self.opts.gp - 1  # Set the end index for this job
                self.model.progress.increment()  # Increment progress counter for this model

                # Define helper functions to handle jump, bypass, and early exit scenarios
                def do_jump(i, jump):
                    return min(jump, abs(cp_end - i))

                def bypass_range(i):
                    if i == 0:
                        return range(c[i], c[i] + 1)
                    else:
                        return range(c[i], c[i] + b[i] + 1)

                def early_exit_range(i):
                    if i == 0:
                        return range(c[i], c[i] + 1)
                    else:
                        return range(c[i], cp_end)

                # Check if flag is enabled and if the current job has a flag set
                if self.opts.flag and self.flag.get(c) != 0 and jump == 0:
                    # If jump is not set and there's a flag for this job, set jump
                    jump = do_jump(c[0] - 1, self.flag.get(c))

                # If jump is set, skip to the next iteration of the loop
                if jump > 0:
                    jump = jump - 1
                    continue

                # Log model progress for each objective
                for o in self.model.iter_obj2:
                    log += f"e{o + 1}: {self.model.e[o, c[o]]}, "
                    self.model.model.e[o + 2] = self.model.e[o, c[o]]

                # Activate objective 0 and solve the model
                self.model.obj_activate(0)
                self.model.solve()
                self.model.models_solved.increment()

                # Check if early exit is enabled and if the model is infeasible
                if self.opts.early_exit and self.model.is_infeasible():
                    # If so, increment infeasibilities counter
                    self.model.infeasibilities.increment()

                    # Set flag if flag is enabled
                    if self.opts.flag:
                        self.flag.set(early_exit_range, self.opts.gp, self.model.iter_obj2)

                    jump = do_jump(c[0], self.opts.gp)  # Set jump
                    log += "infeasible"  # Log infeasibility

                    # Log progress if process logging is enabled
                    if self.opts.process_logging:
                        self.logger.info(log)

                    continue  # Skip to next iteration of loop

                # Calculate slack values and set jump if bypass is enabled
                elif self.opts.bypass and self.model.is_optimal():
                    b = []
                    for i in self.model.iter_obj2:
                        step = self.model.obj_range[i] / (self.opts.gp - 1)
                        slack = round(self.model.slack_val(i + 1))
                        b.append(int(slack / step))

                    # Log jump and set flag if enabled
                    log += f"jump: {b[0]}, "
                    if self.opts.flag:
                        self.flag.set(bypass_range, b[0] + 1, self.model.iter_obj2)
                    jump = do_jump(c[0], b[0])

                # If model is optimal, calculate and log solutions
                sols = []
                if self.model.is_optimal():
                    sols.append(
                        self.model.obj_val(0)
                        - self.opts.eps
                        * sum(
                            10 ** (-1 * (o)) * self.model.slack_val(o + 1) / self.model.obj_range[o]
                            for o in self.model.iter_obj2
                        )
                    )

                    for o in self.model.iter_obj2:
                        sols.append(self.model.obj_val(o + 1))

                    # Put results into queue as a dictionary
                    sols_dict = {tuple(sols): self.model.get_vars()}
                    self.queues.put_result(sols_dict)

                    # Log solutions if process logging is enabled
                    log += f"solutions: {sols}"
                    if self.opts.process_logging:
                        self.logger.info(log)

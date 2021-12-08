import logging
from multiprocessing import Process
from pyaugmecon.flag import Flag
from pyaugmecon.model import Model
from pyaugmecon.options import Options
from pyaugmecon.queue_handler import QueueHandler


class SolverProcess(Process):
    def __init__(
        self, p_num, opts: Options, model: Model, queues: QueueHandler, flag: Flag
    ):
        Process.__init__(self)
        self.p_num = p_num
        self.opts = opts
        self.model = model
        self.queues = queues
        self.flag = flag

    def run(self):
        jump = 0
        if self.opts.process_logging:
            logger = logging.getLogger(self.opts.log_name)
            logger.setLevel(logging.INFO)

        self.model.unpickle()

        while True:
            work = self.queues.get_work(self.p_num)

            if work:
                for c in work:
                    log = f"Process: {self.p_num}, index: {c}, "

                    cp_end = self.opts.gp - 1

                    self.model.progress.increment()

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

                    if self.opts.flag and self.flag.get(c) != 0 and jump == 0:
                        jump = do_jump(c[0] - 1, self.flag.get(c))

                    if jump > 0:
                        jump = jump - 1
                        continue

                    for o in self.model.iter_obj2:
                        log += f"e{o + 1}: {self.model.e[o, c[o]]}, "
                        self.model.model.e[o + 2] = self.model.e[o, c[o]]

                    self.model.obj_activate(0)
                    self.model.solve()
                    self.model.models_solved.increment()

                    if self.opts.early_exit and self.model.is_infeasible():
                        self.model.infeasibilities.increment()
                        if self.opts.flag:
                            self.flag.set(
                                early_exit_range, self.opts.gp, self.model.iter_obj2
                            )
                        jump = do_jump(c[0], self.opts.gp)

                        log += "infeasible"
                        if self.opts.process_logging:
                            logger.info(log)
                        continue
                    elif self.opts.bypass and self.model.is_optimal():
                        b = []

                        for i in self.model.iter_obj2:
                            step = self.model.obj_range[i] / (self.opts.gp - 1)
                            slack = round(self.model.slack_val(i + 1))
                            b.append(int(slack / step))

                        log += f"jump: {b[0]}, "

                        if self.opts.flag:
                            self.flag.set(bypass_range, b[0] + 1, self.model.iter_obj2)
                        jump = do_jump(c[0], b[0])

                    sols = []

                    if self.model.is_optimal():
                        sols.append(
                            self.model.obj_val(0)
                            - self.opts.eps
                            * sum(
                                10 ** (-1 * (o))
                                * self.model.slack_val(o + 1)
                                / self.model.obj_range[o]
                                for o in self.model.iter_obj2
                            )
                        )

                        for o in self.model.iter_obj2:
                            sols.append(self.model.obj_val(o + 1))

                        self.queues.put_result(tuple(sols))

                        log += f"solutions: {sols}"
                        if self.opts.process_logging:
                            logger.info(log)
            else:
                break

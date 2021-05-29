import os
import logging
from pathlib import Path
from pyaugmecon.options import Options

logging.getLogger("pyomo.core").setLevel(logging.ERROR)


class Logs(object):
    def __init__(self, opts: Options):
        self.opts = opts

        if not os.path.exists(self.opts.logdir):
            os.makedirs(self.opts.logdir)

        self.logdir = f"{Path().absolute()}/{self.opts.logdir}/"
        self.logfile = f"{self.logdir}{self.opts.log_name}.log"
        self.logger = logging.getLogger(opts.log_name)
        self.handler = logging.FileHandler(self.logfile)
        self.formatter = logging.Formatter("[%(asctime)s] %(message)s")
        self.handler.setFormatter(self.formatter)
        self.logger.addHandler(self.handler)
        self.handler.setLevel(logging.INFO)

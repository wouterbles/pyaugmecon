import os
import logging
from pathlib import Path
from pyaugmecon.options import Options

logging.getLogger('pyomo.core').setLevel(logging.ERROR)


class Logs(object):
    def __init__(self, name, opts: Options):
        self.name = name
        self.opts = opts

        if not os.path.exists(self.opts.logdir):
            os.makedirs(self.opts.logdir)

        self.logdir = f'{Path().absolute()}/{self.opts.logdir}/'
        self.logfile = f'{self.logdir}{self.name}.log'

        self.logger = logging.getLogger(self.name)

        self.handler = logging.FileHandler(self.logfile)
        self.formatter = logging.Formatter('[%(asctime)s] %(message)s')
        self.handler.setFormatter(self.formatter)
        self.logger.addHandler(self.handler)
        self.handler.setLevel(logging.INFO)

import logging
import os
from pathlib import Path

from pyaugmecon.options import Options

logging.getLogger("pyomo.core").setLevel(logging.ERROR)


class Logs:
    def __init__(self, opts: Options):
        """
        Initializes the Logs object.

        Parameters:
        ----------
        opts: Options
            An object containing the options of the program.

        """
        self.opts = opts

        # Create log directory if it does not exist
        if not os.path.exists(self.opts.logdir):
            os.makedirs(self.opts.logdir)

        # Set the log directory and log file path
        self.logdir = f"{Path().absolute()}/{self.opts.logdir}/"
        self.logfile = f"{self.logdir}{self.opts.log_name}.log"
        self.logger = logging.getLogger(opts.log_name)  # Set up logger object
        self.handler = logging.FileHandler(self.logfile)  # Set up handler object for logger object
        self.formatter = logging.Formatter("[%(asctime)s] %(message)s")  # Set up formatter object for handler object
        self.handler.setFormatter(self.formatter)  # Add formatter object to handler object
        self.logger.addHandler(self.handler)  # Add handler object to logger object
        self.handler.setLevel(logging.INFO)  # Set logging level for the handler object

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
        if not opts.external_logger:
            logfile = f"{self.logdir}{self.opts.log_name}.log"
            logger = logging.getLogger(opts.log_name)  # Set up logger object
            handler = logging.FileHandler(logfile)  # Set up handler object for logger object
            formatter = logging.Formatter("[%(asctime)s] %(message)s")  # Set up formatter object for handler object
            handler.setFormatter(formatter)  # Add formatter object to handler object
            logger.addHandler(handler)  # Add handler object to logger object
            handler.setLevel(logging.INFO)  # Set logging level for the handler object

import itertools
from multiprocessing import Manager

from pyaugmecon.options import Options


class Flag:
    def __init__(self, opts: Options):
        """
        Initialize the Flag class.

        Parameters
        ----------
        opts : Options
            An instance of the Options class.

        """
        self.opts = opts
        self.flag = Manager().dict() if self.opts.shared_flag else {}

    def set(self, flag_range, value, iter):
        """
        Set the specified value for all possible index combinations in the dictionary.

        Parameters
        ----------
        flag_range : function
            A function that returns the range of values that an index can take.
        value : any
            The value to be set.
        iter : iterable
            An iterable containing the indices.

        """
        indices = list(itertools.product(*(flag_range(o) for o in iter)))
        self.flag.update({gp: value for gp in indices})

    def get(self, i):
        """
        Get the value of the flag for the given index combination.

        Parameters
        ----------
        i : tuple
            The index combination.

        Returns
        -------
        any
            The value of the flag for the given index combination or 0 if it doesn't exist.

        """
        return self.flag.get(i, 0)

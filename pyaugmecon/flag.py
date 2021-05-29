import itertools
from multiprocessing import Manager
from pyaugmecon.options import Options


class Flag(object):
    def __init__(self, opts: Options):
        self.opts = opts

        if self.opts.shared_flag:
            self.flag = Manager().dict()
        else:
            self.flag = {}

    def set(self, flag_range, value, iter):
        indices = [tuple([n for n in flag_range(o)]) for o in iter]
        iter = list(itertools.product(*indices))
        tmp_flag = {}

        for gp in iter:
            tmp_flag[gp] = value

        self.flag.update(tmp_flag)

    def get(self, i):
        return self.flag.get(i, 0)

import itertools
import numpy as np
from multiprocessing import shared_memory, Lock
from pyaugmecon.options import Options
from pyaugmecon.model import Model


class Flag(object):
    def __init__(self, model: Model, opts: Options):
        self.opts = opts
        self.lock = Lock()
        self.flag_size = tuple(self.opts.gp for _ in model.iter_obj2)

        a = np.zeros(shape=self.flag_size, dtype=np.int64)
        self.shm = shared_memory.SharedMemory(create=True, size=a.nbytes)
        flag = np.ndarray(a.shape, dtype=np.int64, buffer=self.shm.buf)
        flag[:] = a[:]
        self.shm_name = self.shm.name

    def set(self, flag_range, value, iter):
        shm = shared_memory.SharedMemory(name=self.shm_name)
        flag = np.ndarray(self.flag_size, dtype=np.int64, buffer=shm.buf)
        
        indices = [tuple([n for n in flag_range(o)])
                   for o in iter]
        iter = list(itertools.product(*indices))
        self.lock.acquire()
        for i in iter:
            if value > flag[i]:
                flag[i] = value

        self.lock.release()
        shm.close()

    def get(self, i):
        shm = shared_memory.SharedMemory(name=self.shm_name)
        flag = np.ndarray(self.flag_size, dtype=np.int64, buffer=shm.buf)
        return flag[i]

    def close(self):
        self.shm.close()
        self.shm.unlink()

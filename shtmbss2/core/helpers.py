import multiprocessing as mp
import traceback
import numpy as np

from shtmbss2.common.config import SYMBOLS


class NeuronType:
    class Dendrite:
        ID = 0
        NAME = "dendrite"

    class Soma:
        ID = 1
        NAME = "soma"

    class Inhibitory:
        ID = 2
        NAME = "inhibitory"


class RecTypes:
    SPIKES = "spikes"
    V = "v"


class Process(mp.Process):
    def __init__(self, *args, **kwargs):
        mp.Process.__init__(self, *args, **kwargs)
        self._pconn, self._cconn = mp.Pipe()
        self._exception = None

    def run(self):
        try:
            mp.Process.run(self)
            self._cconn.send(None)
        except Exception as e:
            tb = traceback.format_exc()
            self._cconn.send((e, tb))

    @property
    def exception(self):
        if self._pconn.poll():
            self._exception = self._pconn.recv()
        return self._exception


def symbol_from_label(label, endpoint):
    return label.split('_')[1].split('>')[endpoint]


def id_to_symbol(index):
    return list(SYMBOLS.keys())[index]


def moving_average(a, n=4):
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n

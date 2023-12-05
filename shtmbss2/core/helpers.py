import multiprocessing as mp
import traceback
import numpy as np

from inspect import isclass

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

    @staticmethod
    def get_all_types():
        all_types = list()
        for n_type_name, n_type in NeuronType.__dict__.items():
            if isclass(n_type):
                all_types.append(n_type)
        return all_types


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
            # ToDo: Figure out why the exception doesn't get carried back to top
            print(e.with_traceback())
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
    ret = ret[n - 1:] / n
    if n > 1:
        return np.concatenate((moving_average(a[:n-1], n=n-1), ret), axis=0)
    else:
        return ret


def calculate_trace(start_value, t_start, t_end, t_events, decay_factor):
    trace_new = start_value
    for t_event in t_events:
        if not t_start < t_event < t_end:
            continue
        trace_new = trace_new * np.exp(-(t_event-t_start) / decay_factor) + 1.0
        t_start = t_event

    trace_new = trace_new * np.exp(-(t_end-t_start) / decay_factor)

    return trace_new

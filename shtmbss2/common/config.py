from inspect import isclass

import shtmbss2
import logging
import os
import string
import numpy as np

from os.path import join, dirname, split

# Workaround to remove "Invalid MIT-MAGIC-COOKIE-1 key" error message caused by import of mpi4py in NumpyRNG (pyNN)
os.environ["HWLOC_COMPONENTS"] = "-gl"


PY_PKG_PATH = split(dirname(shtmbss2.__file__))[0]


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


class NamedStorage:
    @classmethod
    def get_all(cls):
        return [v for n, v in vars(cls).items() if not (n.startswith('_') or callable(v))]


class Backends(NamedStorage):
    BRAIN_SCALES_2 = 'bss2'
    NEST = 'nest'


class RunType(NamedStorage):
    MULTI = "multi"
    SINGLE = "single"


class FileType(NamedStorage):
    DATA = 'data'
    FIGURE = 'figure'
    MODEL = 'model'
    OPTIMIZATION = 'optimization'


class ExperimentType(NamedStorage):
    EVAL_SINGLE = 'eval-single'
    EVAL_MULTI = 'eval-multi'
    OPT_GRID = 'opt-grid'
    INSTANCE = 'instance'


class PerformanceType(NamedStorage):
    ALL_SYMBOLS = "all_symbols"
    LAST_SYMBOL = "last_symbol"


class PerformanceMetrics(NamedStorage):
    ERROR = 'error'
    FP = 'false_positive'
    FN = 'false_negative'
    ACTIVE_SOMAS = 'active_somas'
    # ACTIVE_DENDRITES = 'active_dendrite'


class StatisticalMetrics(NamedStorage):
    MEAN = 'mean'
    STD = 'std'
    MEDIAN = 'median'
    PERCENTILE = 'percentile'


class LogHandler(NamedStorage):
    FILE = 0
    STREAM = 1


class RuntimeConfig(NamedStorage):
    backend = None
    backend_initialization = False
    config_prefix = "shtm2bss_config"
    instance_digits = 2
    saved_weights = ["exc_to_exc", "exc_to_inh"]
    saved_events = [NeuronType.Soma, NeuronType.Dendrite, NeuronType.Inhibitory]
    saved_network_vars = ["trace_dendrites"]
    saved_plasticity_vars = ["permanence", "permanence_min", "permanences", "weights", "x"]
    saved_instance_params = []


# Logging
class Log(NamedStorage):
    FILE = join(PY_PKG_PATH, 'data/log/shtm2bss.log')
    # FORMAT_FILE = "[%(asctime)s] [%(filename)s:%(lineno)s - %(funcName)20s() ] [%(levelname)-8s] %(message)s"
    FORMAT_FILE = "[%(asctime)s] [%(filename)-20s:%(lineno)-4s] [%(levelname)-8s] %(message)s"
    FORMAT_SCREEN = "%(log_color)s%(message)s"
    LEVEL_FILE = logging.INFO
    LEVEL_SCREEN = logging.INFO
    DATEFMT = '%d.%m.%Y %H:%M:%S'


PATH_CONFIG = join(PY_PKG_PATH, 'config')
PATH_MODELS = join(PY_PKG_PATH, 'models')

EXPERIMENT_FOLDERS = {
    Backends.NEST: join(PY_PKG_PATH, 'data/evaluation/nest'),
    Backends.BRAIN_SCALES_2: join(PY_PKG_PATH, 'data/evaluation/bss2')
}
EXPERIMENT_SUBFOLDERS = {
    ExperimentType.EVAL_SINGLE: 'single',
    ExperimentType.EVAL_MULTI: 'multi',
    ExperimentType.OPT_GRID: 'grid'
}
EXPERIMENT_SETUP_FILE_NAME = {
    ExperimentType.EVAL_SINGLE: 'experiments_single.csv',
    ExperimentType.EVAL_MULTI: 'experiments_multi.csv',
    ExperimentType.OPT_GRID: 'experiments_grid.csv',
    ExperimentType.INSTANCE: 'experimental_results.csv'
}

SYMBOLS = {symbol: index for index, symbol in enumerate(string.ascii_uppercase)}

NP_STATISTICS = {m: getattr(np, m) for m in StatisticalMetrics.get_all()}

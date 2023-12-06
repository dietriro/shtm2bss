import shtmbss2
import logging
import os
import string

from os.path import join, dirname, split


# Workaround to remove "Invalid MIT-MAGIC-COOKIE-1 key" error message caused by import of mpi4py in NumpyRNG (pyNN)
os.environ["HWLOC_COMPONENTS"] = "-gl"


PY_PKG_PATH = split(dirname(shtmbss2.__file__))[0]


class Backends:
    BRAIN_SCALES_2 = 'bss2'
    NEST = 'nest'


class RunType:
    MULTI = "multi"
    SINGLE = "single"


class FileType:
    DATA = 'data'
    FIGURE = 'figure'
    MODEL = 'model'
    OPTIMIZATION = 'optimization'


class ExperimentType:
    EVAL_SINGLE = 'eval_single'
    EVAL_MULTI = 'eval_multi'
    INSTANCE = 'instance'


class PerformanceType:
    ALL_SYMBOLS = "all_symbols"
    LAST_SYMBOL = "last_symbol"


class LogHandler:
    FILE = 0
    STREAM = 1


class RuntimeConfig:
    backend = None
    config_prefix = "shtm2bss_config"
    saved_network_vars = ["exc_to_exc", "exc_to_inh"]
    saved_plasticity_vars = ["permanence", "permanence_min", "permanences", "weights", "x", "z"]
    saved_instance_params = ["Experiment.type", "Experiment.id", "Experiment.sequences", "Experiment.runtime",
                             "Experiment.episodes"]


# Logging
class Log:
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
    FileType.DATA: 'data',
    FileType.FIGURE: 'figures',
    FileType.MODEL: 'models',
    FileType.OPTIMIZATION: 'optimizations'
}
EXPERIMENT_SETUP_FILE_NAME = {
    ExperimentType.EVAL_SINGLE: 'experiments_single.csv',
    ExperimentType.EVAL_MULTI: 'experiments_multi.csv',
    ExperimentType.INSTANCE: 'experimental_results.csv'
}

SYMBOLS = {symbol: index for index, symbol in enumerate(string.ascii_uppercase)}

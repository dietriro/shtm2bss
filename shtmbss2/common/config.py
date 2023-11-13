import shtmbss2
import logging
import os

from os.path import join, dirname, split


# Workaround to remove "Invalid MIT-MAGIC-COOKIE-1 key" error message caused by import of mpi4py in NumpyRNG (pyNN)
os.environ["HWLOC_COMPONENTS"] = "-gl"


PY_PKG_PATH = split(dirname(shtmbss2.__file__))[0]


class Backends:
    BRAIN_SCALES_2 = 'bss2'
    NEST = 'nest'


class PlotConfig:
    FILE_TYPE = 'pdf'


class LogHandler:
    FILE = 0
    STREAM = 1


class RuntimeConfig:
    backend = None
    config_prefix = "shtm2bss_config"


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

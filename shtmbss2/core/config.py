import shtmbss2
import logging

from os.path import join, dirname, split


PY_PKG_PATH = split(dirname(shtmbss2.__file__))[0]


class PlotConfig:
    FILE_TYPE = 'pdf'


class LogHandler:
    FILE = 0
    STREAM = 1


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
CONFIG_PREFIX = "shtm2bss_config"

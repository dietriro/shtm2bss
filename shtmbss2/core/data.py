import yaml

from shtmbss2.core.config import *
from shtmbss2.core.logging import log


def load_yaml(path_yaml, file_name_yaml):
    with open(join(path_yaml, file_name_yaml)) as file:
        try:
            data = yaml.safe_load(file)
        except yaml.YAMLError as exc:
            log.exception(exc)

    return data


def load_config(network_type):
    config_file_name = f"{CONFIG_PREFIX}_{type(network_type).__name__}.yaml"
    return load_yaml(PATH_CONFIG, config_file_name)



from shtmbss2.common.config import *

RuntimeConfig.backend = Backends.NEST

RuntimeConfig.config_prefix = "shtm2bss_config_nest"

DEFAULT_MC_MODEL_FILE = join(PATH_MODELS, "mc", "iaf_psc_exp_nonlineardendrite.nestml")
import sys
import os

pkg_path = os.path.abspath(os.path.join(os.getcwd(), os.pardir))
sys.path.append(pkg_path)

import warnings
import numpy as np
import matplotlib.pyplot as plt

from shtmbss2.nest.network import SHTMTotal
from shtmbss2.common.config import NeuronType
from shtmbss2.common.config import *
from shtmbss2.core.logging import log

np.set_printoptions(threshold=np.inf, suppress=True, linewidth=np.inf)
warnings.filterwarnings(action='ignore', category=UserWarning)

# log.handlers[LogHandler.STREAM].setLevel(logging.DEBUG)
# log.handlers[LogHandler.FILE].setLevel(logging.DETAIL)


### Config

# pynn.logger.default_config(level=pynn.logger.LogLevel.DEBUG)
v_rest_calib = 275
num_sim_steps = 150
debug = False


### Initialize

shtm = SHTMTotal()

shtm.init_neurons()
shtm.init_connections(debug=debug)
shtm.init_external_input()

# shtm.set_weights_exc_exc(10, 1, post_ids=[3,4,5], p_con=1.0)

### Run


# shtm.set_weights_exc_exc(10, 1, post_ids=[3,4,5], p_con=1.0)
shtm.run(steps=20, plasticity_enabled=True)

# shtm.plot_v_exc(alphabet_range=[2], neuron_range="all", neuron_type=NeuronType.Soma)


### Plot
# shtm.plot_events(neuron_types="all", size=(24, 10))

# shtm.plot_v_exc(alphabet_range=[0], neuron_range=range(3), neuron_type=NeuronType.Inhibitory, show_legend=True)

# shtm.plot_permanence_history(plot_con_ids=[1, 5])

# shtm.plot_permanence_diff()

# shtm.plot_performance()

# shtm.plot_permanence_history(plot_con_ids=[0, 1])


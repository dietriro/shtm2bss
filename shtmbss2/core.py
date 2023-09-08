import numpy as np
import copy
import pynn_brainscales.brainscales2 as pynn

from _static.common.helpers import setup_hardware_client
from _static.common.helpers import get_nightly_calibration


def hardware_initialization(refractory_clock_scale):
    # Setup BrainScaleS-2
    setup_hardware_client()

    # Load most recent calibration for board
    calib = get_nightly_calibration()

    # Set refractory clock scale - for what?
    for backend in calib.neuron_block.backends:
        backend.clock_scale_fast = refractory_clock_scale

    # ToDo: Still needed?
    period_refractory_clock = 1. / (250e6 / 2 ** (refractory_clock_scale + 1))
    print(period_refractory_clock)

    # setup PyNN and inject calibration data
    pynn.setup(initial_config=calib)



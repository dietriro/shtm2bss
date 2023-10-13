from shtmbss2.core.config import *
from shtmbss2.core.logging import log
from shtmbss2.core.data import load_config


class Parameters:
    class Experiment:
        type = None
        sequences = None
        runtime = None
        episodes = None
        run_add_calib = None

    class Plotting:
        size = None
        file_type = None
        save_figure = None

    class Network:
        num_symbols = None
        num_neurons = None

    class Encoding:
        dt_stm = None
        dt_seq = None

    class Plasticity:
        type = None
        learning_factor = None
        permanence_max = None
        threshold = None
        y = None
        lambda_plus = None
        lambda_minus = None
        lambda_h = None
        target_rate_h = None
        tau_plus = None
        tau_h = None
        delta_t_min = None
        delta_t_max = None
        dt = None

    class Neurons:
        class Inhibitory:
            v_rest = None
            v_reset = None
            v_thresh = None
            tau_m = None
            tau_syn_I = None
            tau_syn_E = None
            tau_refrac = None

        class Excitatory:
            v_rest = None
            v_reset = None
            v_thresh = None
            tau_m = None
            tau_syn_I = None
            tau_syn_E = None
            tau_refrac = None

    class Synapses:
        w_ext_exc = None
        w_exc_exc = None
        w_exc_inh = None
        w_inh_exc = None
        p_exc_exc = None

    class Calibration:
        v_rest_calib = None

    def __init__(self, network_type, custom_params=None):
        self.load_default_params(network_type)

        log.debug(f"Successfully loaded parameters for '{network_type}'")

        # Set specific parameters loaded from individual configuration
        if custom_params is not None:
            for name, value in custom_params.items():
                category_objs = name.split('.')
                category_obj = self
                for category_name in category_objs[:-1]:
                    category_obj = getattr(category_obj, category_name)
                setattr(category_obj, name, value)

        log.debug(f"Successfully set custom parameters for '{network_type}'")

    def load_default_params(self, network_type):
        default_params = load_config(network_type)
        self.set_params(self, default_params)

    def set_params(self, category_obj, parameters):
        for name, value in parameters.items():
            if type(value) is dict:
                if hasattr(category_obj, name.capitalize()):
                    self.set_params(getattr(category_obj, name.capitalize()), value)
                else:
                    log.warn(f"'{category_obj}' does not have an object '{name.capitalize()}'")
                    continue
            else:
                if hasattr(category_obj, name):
                    setattr(category_obj, name, value)

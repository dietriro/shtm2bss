from abc import ABC

from shtmbss2.core.logging import log
from shtmbss2.core.data import load_config


class ParameterGroup:
    @classmethod
    def dict(cls):
        p_dict_original = cls.__dict__
        p_dict = {v: p_dict_original[v] for v, m in vars(cls).items() if not (v.startswith('_') or callable(m))}
        return p_dict


class Parameters:
    class Experiment(ParameterGroup):
        type = None
        sequences = None
        seq_repetitions = None
        runtime = None
        episodes = None
        run_add_calib = None

    class Plotting(ParameterGroup):
        size = None
        file_type = None
        save_figure = None

    class Network(ParameterGroup):
        num_symbols = None
        num_neurons = None

    class Backend(ParameterGroup):
        module_name = None
        neuron_name = None

    class Encoding(ParameterGroup):
        dt_stm = None
        dt_seq = None

    class Plasticity(ParameterGroup):
        type = None
        learning_factor = None
        permanence_max = None
        threshold = None
        w_mature = None
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
        class Inhibitory(ParameterGroup):
            c_m = None
            v_rest = None
            v_reset = None
            v_thresh = None
            tau_m = None
            tau_syn_I = None
            tau_syn_E = None
            tau_refrac = None

        class Excitatory(ParameterGroup):
            c_m = None
            v_rest = None
            v_reset = None
            v_thresh = None
            tau_m = None
            tau_syn_I = None
            tau_syn_E = None
            tau_syn_ext = None
            tau_syn_den = None
            tau_syn_inh = None
            tau_refrac = None

        class Dendrite(ParameterGroup):
            I_p = None
            tau_dAP = None
            theta_dAP = None

    class Synapses(ParameterGroup):
        dyn_inh_weights = None
        w_ext_exc = None
        w_exc_exc = None
        w_exc_inh = None
        w_inh_exc = None
        p_exc_exc = None
        receptor_ext_exc = None
        receptor_exc_exc = None
        receptor_exc_inh = None
        receptor_inh_exc = None

    class Calibration(ParameterGroup):
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

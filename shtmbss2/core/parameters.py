import inspect
from abc import ABC

from shtmbss2.core.logging import log
from shtmbss2.core.data import load_config


class ParameterGroup:
    @classmethod
    def dict(cls, exclude_none=False):
        p_dict_original = cls.__dict__
        p_dict = dict()
        for v_name, v_instance in vars(cls).items():
            if not (v_name.startswith('_') or inspect.isfunction(v_instance)):
                if inspect.isclass(v_instance):
                    p_dict[v_name] = v_instance.dict(exclude_none=exclude_none)
                else:
                    if exclude_none and p_dict_original[v_name] is None:
                        continue
                    p_dict[v_name] = p_dict_original[v_name]
        return p_dict


class Parameters(ParameterGroup):
    class Experiment(ParameterGroup):
        type: str = None
        id: str = None
        sequences: list = None
        seq_repetitions: int = None
        runtime: float = None
        episodes: int = None
        run_add_calib: bool = None
        autosave: bool = None
        autosave_epoches: int = None

    class Plotting(ParameterGroup):
        size: list = None
        file_type: str = None
        save_figure: bool = None

    class Performance(ParameterGroup):
        compute_performance: bool = None
        method: str = None

    class Network(ParameterGroup):
        num_symbols: int = None
        num_neurons: int = None
        pattern_size: int = None

    class Backend(ParameterGroup):
        module_name: str = None
        neuron_name: str = None

    class Encoding(ParameterGroup):
        dt_stm: float = None
        dt_seq: float = None
        t_exc_start: float = None

    class Plasticity(ParameterGroup):
        type: str = None
        learning_factor: float = None
        permanence_init_min: float = None
        permanence_init_max: float = None
        permanence_max: float = None
        threshold: float = None
        w_mature: float = None
        y: float = None
        lambda_plus: float = None
        lambda_minus: float = None
        lambda_h: float = None
        target_rate_h: float = None
        tau_plus: float = None
        tau_h: float = None
        delta_t_min: float = None
        delta_t_max: float = None
        dt: float = None

    class Neurons(ParameterGroup):
        class Inhibitory(ParameterGroup):
            c_m: float = None
            v_rest: float = None
            v_reset: float = None
            v_thresh: float = None
            tau_m: float = None
            tau_syn_I: float = None
            tau_syn_E: float = None
            tau_refrac: float = None

        class Excitatory(ParameterGroup):
            c_m: float = None
            v_rest: float = None
            v_reset: float = None
            v_thresh: float = None
            tau_m: float = None
            tau_syn_I: float = None
            tau_syn_E: float = None
            tau_syn_ext: float = None
            tau_syn_den: float = None
            tau_syn_inh: float = None
            tau_refrac: float = None

        class Dendrite(ParameterGroup):
            I_p: float = None
            tau_dAP: float = None
            theta_dAP: float = None

    class Synapses(ParameterGroup):
        dyn_inh_weights: bool = None
        w_ext_exc: float = None
        w_exc_exc: float = None
        w_exc_inh: float = None
        w_inh_exc: float = None
        p_exc_exc: float = None
        receptor_ext_exc: str = None
        receptor_exc_exc: str = None
        receptor_exc_inh: str = None
        receptor_inh_exc: str = None
        delay_ext_exc: float = None
        delay_exc_exc: float = None
        delay_exc_inh: float = None
        delay_inh_exc: float = None

    class Calibration(ParameterGroup):
        v_rest_calib: float = None

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


import inspect
from abc import ABC

from shtmbss2.common.config import *
from shtmbss2.core.logging import log
from shtmbss2.core.data import load_config, get_experiment_folder, load_yaml


class ParameterGroup:
    _to_evaluate: list = list()

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

    @classmethod
    def evaluate(cls, recursive=True):
        for param_name in cls._to_evaluate:
            if hasattr(cls, param_name):
                value = getattr(cls, param_name)
                try:
                    value = eval(value)
                except Exception as e:
                    log.warning(f"Could not evaluate parameter {param_name}.")
                    log.warning(e)
                    value = None
                setattr(cls, param_name, value)
            else:
                log.warning(f"Could not find parameter {param_name} for class {cls.__str__}")

        if recursive:
            for v_name, v_instance in vars(cls).items():
                if not (v_name.startswith('_') or inspect.isfunction(v_instance)):
                    if inspect.isclass(v_instance):
                        v_instance.evaluate(recursive=recursive)


class Parameters(ParameterGroup):
    def __init__(self, network_type):
        self.network_type = network_type
        self.config_type = None

    def load_default_params(self, custom_params=None):
        default_params = load_config(self.network_type, config_type=self.config_type)
        self.set_params(self, default_params)

        log.debug(f"Successfully loaded parameters for '{self.network_type}'")

        # Set specific parameters loaded from individual configuration
        if custom_params is not None:
            for name, value in custom_params.items():
                category_objs = name.split('.')
                category_obj = self
                for category_name in category_objs[:-1]:
                    category_obj = getattr(category_obj, category_name)
                setattr(category_obj, category_objs[-1], value)

        log.debug(f"Successfully set custom parameters for '{self.network_type}'")

    def load_experiment_params(self, experiment_type, experiment_id, experiment_num, experiment_subnum=None,
                               instance_id=None):
        if ((experiment_type == ExperimentType.EVAL_MULTI or experiment_type == ExperimentType.OPT_GRID_MULTI)
                and instance_id is None):
            instance_id = 0

        experiment_folder_path = get_experiment_folder(self.network_type, experiment_type, experiment_id,
                                                       experiment_num, experiment_subnum=experiment_subnum,
                                                       instance_id=instance_id)

        saved_params = load_yaml(experiment_folder_path, f"config_{self.config_type}.yaml")

        self.set_params(self, saved_params)

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


class NetworkParameters(Parameters):
    class Experiment(ParameterGroup):
        type: str = None
        id: str = None
        sequences: list = None
        seq_repetitions: int = None
        runtime: float = None
        episodes: int = None
        run_add_calib: bool = None
        save_final: bool = None
        save_auto: bool = None
        save_auto_epoches: int = None
        generate_rand_seed_offset: bool = None
        seed_offset: int = None
        log_weights: bool = None
        log_permanence: bool = None

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
        t_scaling_factor: float = None
        num_repetitions: int = None

    class Plasticity(ParameterGroup):
        type: str = None
        execution_start: float = None
        execution_interval: float = None
        learning_factor: float = None
        weight_learning: bool = None
        weight_learning_scale: float = None
        permanence_init_min: float = None
        permanence_init_max: float = None
        permanence_max: float = None
        permanence_threshold: float = None
        correlation_threshold: int = None
        w_mature: float = None
        y: float = None
        lambda_plus: float = None
        lambda_minus: float = None
        lambda_h: float = None
        homeostasis_depression_rate: float = None
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
        dyn_weight_calculation: bool = None
        w_exc_inh_dyn: float = None
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
        j_ext_exc_psp: float = None
        j_exc_inh_psp: float = None
        j_inh_exc_psp: float = None
        _to_evaluate: list = ["j_ext_exc_psp",
                              "j_exc_inh_psp",
                              "j_inh_exc_psp"]

    class Calibration(ParameterGroup):
        v_rest_calib: float = None
        padi_bus_dacen_extension = None
        correlation_amplitude = None
        correlation_time_constant = None

    def __init__(self, network_type):
        super().__init__(network_type)

        self.config_type = ConfigType.NETWORK


class PlottingParametersBase(ParameterGroup):
    size: list = None
    dpi: int = None
    line_width: int = None

    class Fontsize(ParameterGroup):
        title: int = None
        legend: int = None
        axis_labels: int = None
        tick_labels: int = None


class PlottingParameters(Parameters):
    class Performance(PlottingParametersBase):
        pass

    def __init__(self, network_type):
        super().__init__(network_type)

        self.config_type = ConfigType.PLOTTING



import warnings
import numpy as np
import itertools

from shtmbss2.common.config import *
from shtmbss2.core.data import load_config, get_last_experiment_num
from shtmbss2.core.logging import log

np.set_printoptions(threshold=np.inf, suppress=True, linewidth=np.inf)
warnings.filterwarnings(action='ignore', category=UserWarning)

log.handlers[LogHandler.STREAM].setLevel(logging.ESSENS)


class GridSearch:
    def __init__(self, model_type, experiment_id):
        self.model_type = model_type
        self.experiment_id = experiment_id

        self.experiment_type = ExperimentType.OPT_GRID
        self.experiment_num = None
        self.config = None

        self.load_config()

    def load_config(self):
        self.config = load_config(self.model_type, self.experiment_type)

    def __run_experiment(self, parameters, experiment_num, instance_id, steps=None):
        model = self.model_type(experiment_type=ExperimentType.OPT_GRID, instance_id=instance_id, seed_offset=0,
                                **parameters)

        # set save_auto to false in order to minimize file lock timeouts
        model.p.Experiment.save_auto = False
        model.p.Experiment.save_final = False
        model.experiment_num = experiment_num

        model.init_neurons()
        model.init_connections(debug=False)
        model.init_external_input()

        if RuntimeConfig.backend == Backends.BRAIN_SCALES_2:
            model.init_rec_exc()

        model.run(steps=steps, plasticity_enabled=True, run_type=RunType.SINGLE)

        model.save_full_state(save_basic_data=experiment_num == 0, running_avg_perc=0.5)

        return model

    def run(self, steps=None):
        parameter_names = list()
        parameter_values = list()
        for parameter_name, parameter_config in self.config["parameters"].items():
            if not parameter_config["enabled"]:
                continue
            parameter_names.append(parameter_name)
            parameter_values.append(np.arange(start=parameter_config["min"], stop=parameter_config["max"],
                                              step=parameter_config["step"], dtype=parameter_config["dtype"]).tolist())

        parameter_combinations = list(itertools.product(*parameter_values))

        # retrieve experiment num for new experiment
        self.experiment_num = get_last_experiment_num(self.model_type, self.experiment_id, self.experiment_type) + 1

        log.essens(f"Starting grid-search for {len(parameter_combinations)} parameter combinations")

        for run_i, parameter_combination in enumerate(parameter_combinations):
            log.essens(f"Starting grid-search run {run_i + 1}/{len(parameter_combinations)} for {parameter_combination}")
            parameters = {p_name: p_value for p_name, p_value in zip(parameter_names, parameter_combination)}

            model = self.__run_experiment(parameters, experiment_num=self.experiment_num, instance_id=run_i,
                                          steps=steps)
            performance = model.performance.get_performance_dict(final_result=True, running_avg_perc=0.5)

            log.essens(f"Finished grid-search run {run_i + 1}/{len(parameter_combinations)}")
            log.essens(f"\tParameters: {parameter_combination}")
            log.essens(f"\tPerformance: {performance}")

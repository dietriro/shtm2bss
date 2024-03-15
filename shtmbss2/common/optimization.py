import warnings
import numpy as np
import itertools
import yaml

from shtmbss2.core.helpers import Process
from shtmbss2.common.config import *
from shtmbss2.core.data import load_config, get_last_experiment_num, get_experiment_folder, get_last_instance
from shtmbss2.core.logging import log

np.set_printoptions(threshold=np.inf, suppress=True, linewidth=np.inf)
warnings.filterwarnings(action='ignore', category=UserWarning)


class GridSearch:
    def __init__(self, model_type, experiment_id, experiment_num=None):
        self.model_type = model_type
        self.experiment_id = experiment_id
        self.experiment_num = experiment_num

        self.experiment_type = ExperimentType.OPT_GRID
        self.continuation_id = None
        self.config = None

        self.load_config()

    def load_config(self):
        self.config = load_config(self.model_type, self.experiment_type)

    def save_config(self):
        folder_path_experiment = get_experiment_folder(self.model_type, self.experiment_type, self.experiment_id,
                                                       self.experiment_num, instance_id=None)
        config_file_name = f"config_{self.experiment_type}.yaml"
        file_path = join(folder_path_experiment, config_file_name)
        with open(file_path, 'w') as file:
            yaml.dump(self.config, file)

    def __run_experiment(self, parameters, experiment_num, instance_id, steps=None,
                         optimized_parameter_ranges=None):
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

        model.save_full_state(running_avg_perc=0.5, optimized_parameter_ranges=optimized_parameter_ranges)

    def run(self, steps=None):
        log.handlers[LogHandler.STREAM].setLevel(logging.ESSENS)

        parameter_names = list()
        parameter_values = list()
        parameter_ranges = dict()
        for parameter_name, parameter_config in self.config["parameters"].items():
            if not parameter_config["enabled"]:
                continue
            if ("values" in parameter_config.keys() and parameter_config["values"] is not None
                    and type(parameter_config["values"]) is list and len(parameter_config["values"]) > 0):
                parameter_values.append(parameter_config["values"])
                parameter_ranges[f"Optimized-params.{parameter_name}"] = parameter_config["values"]
            else:
                parameter_values.append(np.arange(start=parameter_config["min"],
                                                  stop=parameter_config["max"]+parameter_config["step"],
                                                  step=parameter_config["step"],
                                                  dtype=parameter_config["dtype"]).tolist())
                parameter_ranges[f"Optimized-params.{parameter_name}"] = (parameter_config["min"],
                                                                          parameter_config["max"],
                                                                          parameter_config["step"])
            parameter_names.append(parameter_name)
        parameter_combinations = list(itertools.product(*parameter_values))
        num_combinations = len(parameter_combinations)

        # set number of digits for file saving/loading into instance folders
        RuntimeConfig.instance_digits = len(str(num_combinations))

        # retrieve experiment num for new experiment
        last_experiment_num = get_last_experiment_num(self.model_type, self.experiment_id, self.experiment_type)
        if self.experiment_num is None:
            self.experiment_num = last_experiment_num + 1

        if self.experiment_num <= last_experiment_num:
            self.continuation_id = get_last_instance(self.model_type, self.experiment_type, self.experiment_id,
                                                     self.experiment_num)
            log.essens(f"Continuing grid-search for {num_combinations-self.continuation_id} "
                       f"parameter combinations of {parameter_names}")
        else:
            log.essens(f"Starting grid-search for {num_combinations} parameter combinations "
                       f"of {parameter_names}")


        for run_i, parameter_combination in enumerate(parameter_combinations):
            if self.continuation_id is not None:
                if run_i < self.continuation_id:
                    continue
                elif run_i == self.continuation_id:
                    log.info(f"Skipped {run_i} parameter combinations. "
                             f"Continuing evaluation with combination {run_i} now.")

            log.essens(f"Starting grid-search run {run_i + 1}/{num_combinations} "
                       f"for {parameter_combination}")
            parameters = {p_name: p_value for p_name, p_value in zip(parameter_names, parameter_combination)}

            p = Process(target=self.__run_experiment, args=(parameters, self.experiment_num, run_i, steps,
                                                            parameter_ranges))
            p.start()
            p.join()

            log.essens(f"Finished grid-search run {run_i + 1}/{num_combinations}")
            log.essens(f"\tParameters: {parameter_combination}")

            if run_i == 0:
                self.save_config()

        log.handlers[LogHandler.STREAM].setLevel(logging.INFO)

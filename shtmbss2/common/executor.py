import time
import warnings
import numpy as np
import multiprocessing as mp

from shtmbss2.core.helpers import Process
from shtmbss2.common.config import *
from shtmbss2.core.logging import log
from shtmbss2.core.data import get_last_experiment_num

if RuntimeConfig.backend == Backends.BRAIN_SCALES_2:
    from shtmbss2.brainscales2.network import SHTMTotal
elif RuntimeConfig.backend == Backends.NEST:
    from shtmbss2.nest.network import SHTMTotal
else:
    raise Exception(f"Backend {RuntimeConfig.backend} not implemented yet. "
                    f"Please choose among [{Backends.BRAIN_SCALES_2}, {Backends.NEST}]")

np.set_printoptions(threshold=np.inf, suppress=True, linewidth=np.inf)
warnings.filterwarnings(action='ignore', category=UserWarning)


class ParallelExecutor:
    def __init__(self, num_instances, experiment_id):
        self.num_instances = num_instances
        self.experiment_id = experiment_id

        self.experiment_type = ExperimentType.EVAL_MULTI
        self.experiment_num = None

    @staticmethod
    def __run_experiment(process_id, file_save_status, lock, experiment_num, seed_offset, steps=None):
        shtm = SHTMTotal(experiment_type=ExperimentType.EVAL_MULTI, instance_id=process_id, seed_offset=seed_offset)

        # set save_auto to false in order to minimize file lock timeouts
        shtm.p.Experiment.save_auto = False
        shtm.p.Experiment.save_final = False
        shtm.experiment_num = experiment_num

        lock.acquire(block=True)
        shtm.init_neurons()
        lock.release()

        shtm.init_connections(debug=False)
        shtm.init_external_input()

        shtm.run(steps=steps, plasticity_enabled=True, run_type=RunType.SINGLE)

        # wait until it's this processes turn to save data (order)
        while process_id > 0 and file_save_status[process_id-1] < 1:
            time.sleep(0.1)

        lock.acquire(block=True)
        shtm.save_full_state()
        lock.release()

        # signal other processes, that this process has finished the data saving process
        file_save_status[process_id] = 1

    def run(self):

        lock = mp.Lock()
        file_save_status = mp.Array("i", [0 for _ in range(self.num_instances)])

        # retrieve experiment num for new experiment
        self.experiment_num = get_last_experiment_num(SHTMTotal, self.experiment_id, self.experiment_type) + 1

        seed_offset = int(time.time())

        log.handlers[LogHandler.STREAM].setLevel(logging.ESSENS)

        processes = []
        for i_inst in range(self.num_instances):
            log.essens(f'Starting network {i_inst}')
            processes.append(Process(target=self.__run_experiment, args=(i_inst, file_save_status, lock,
                                                                         self.experiment_num, seed_offset)))
            processes[i_inst].start()

        for i_inst in range(self.num_instances):
            processes[i_inst].join()

            log.essens(f"Finished simulation {i_inst + 1}/{self.num_instances}")

        log.handlers[LogHandler.STREAM].setLevel(logging.INFO)

        return self.experiment_num

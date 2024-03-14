import os.path

import numpy as np
import matplotlib.pyplot as plt
from abc import ABC, abstractmethod

from shtmbss2.common.config import *
from shtmbss2.core.logging import log
from shtmbss2.core.parameters import Parameters
from shtmbss2.core.helpers import moving_average
from shtmbss2.common.config import NeuronType
from shtmbss2.core.data import get_experiment_folder


class Performance(ABC):
    def __init__(self, parameters, performance_type='single'):
        """

        :param parameters:
        :type parameters: Parameters
        """
        self.p = parameters

        self.type = performance_type
        self.data = None

    @abstractmethod
    def init_data(self):
        pass

    @abstractmethod
    def get_statistic(self, statistic, metric, episode='all', percentile=None):
        pass

    @abstractmethod
    def get_all(self, metric, sequence_id=None):
        pass

    @abstractmethod
    def get_performance_dict(self, final_result=False):
        pass

    @abstractmethod
    def get_data_size(self):
        pass

    @abstractmethod
    def add_data_point(self, data_point, metric, sequence_id):
        pass

    def compute(self, neuron_events, method=PerformanceType.ALL_SYMBOLS):
        log.info(f"Computing performance for {len(self.p.Experiment.sequences)} Sequences.")

        ratio_fp_activation = 0.5
        ratio_fn_activation = 0.5

        t_min = self.p.Encoding.t_exc_start

        for i_seq, seq in enumerate(self.p.Experiment.sequences):
            seq_performance = {metric: list() for metric in PerformanceMetrics.get_all()}

            t_min += i_seq * self.p.Encoding.dt_seq + i_seq * self.p.Encoding.dt_stm

            for i_element, element in enumerate(seq[1:]):
                if method == PerformanceType.LAST_SYMBOL and i_element < len(seq) - 2:
                    continue

                # define min/max for time window of spikes
                if i_element > 0:
                    t_min += self.p.Encoding.dt_stm
                t_max = t_min + self.p.Encoding.dt_stm

                log.debug(f"t_min = {t_min},  t_max = {t_max}")

                # calculate target vector
                output = np.zeros(self.p.Network.num_symbols)
                target = np.zeros(self.p.Network.num_symbols)
                target[SYMBOLS[element]] = 1

                num_dAPs = np.zeros(self.p.Network.num_symbols)
                num_som_spikes = np.zeros(self.p.Network.num_symbols)
                counter_correct = 0

                t_min_next = t_min + self.p.Encoding.dt_stm
                t_max_next = t_max + self.p.Encoding.dt_stm

                for i_symbol in range(self.p.Network.num_symbols):
                    # get dAP's per subpopulation
                    num_dAPs[i_symbol] = np.sum([t_min < item < t_max for sublist in
                                                 neuron_events[NeuronType.Dendrite][i_symbol] for item in
                                                 sublist])

                    # get somatic spikes per subpopulation
                    num_som_spikes[i_symbol] = np.sum([t_min_next < item < t_max_next for sublist in
                                                       neuron_events[NeuronType.Soma][i_symbol] for item in
                                                       sublist])

                    if (i_symbol != SYMBOLS[element] and
                            num_dAPs[i_symbol] >= (ratio_fp_activation * self.p.Network.pattern_size)):
                        output[i_symbol] = 1
                    elif (i_symbol == SYMBOLS[element] and
                          num_dAPs[i_symbol] >= (ratio_fn_activation * self.p.Network.pattern_size)):
                        counter_correct += 1
                        output[i_symbol] = 1

                # calculate Euclidean distance between output and target vector
                # determine prediction error, FP and FN
                error = np.sqrt(sum((output - target) ** 2))
                false_positive = sum(np.heaviside(output - target, 0))
                false_negative = sum(np.heaviside(target - output, 0))

                seq_performance[PerformanceMetrics.ERROR].append(error)
                seq_performance[PerformanceMetrics.FP].append(false_positive)
                seq_performance[PerformanceMetrics.FN].append(false_negative)
                seq_performance[PerformanceMetrics.ACTIVE_SOMAS].append(num_som_spikes[SYMBOLS[element]])
                # seq_performance[PerformanceMetrics.ACTIVE_DENDRITES].append(num_dAPs[SYMBOLS[element]])

            for metric in PerformanceMetrics.get_all():
                self.add_data_point(np.mean(seq_performance[metric]), metric, sequence_id=i_seq)

    def plot(self, statistic, sequences="mean"):
        fig, axs = plt.subplots(1, 3, figsize=[12, 5])

        axs[0].set_ylabel("Prediction error")
        axs[0].set_xlabel("# Training Episodes")

        axs[1].set_ylabel("Rel. frequency")
        axs[1].set_xlabel("# Training Episodes")

        target_pattern_size_line = [self.p.Network.pattern_size / self.p.Network.num_neurons
                                    for _ in range(self.get_data_size())]
        axs[2].plot(target_pattern_size_line, linestyle="dashed", color=f"grey",
                    label=f"Target pattern size ({self.p.Network.pattern_size})")
        axs[2].set_ylabel("Rel. no. of active neurons")
        axs[2].set_xlabel("# Training Episodes")

        fig.tight_layout()

        return fig, axs

    def plot_seq(self, axs, perf_errors, perf_fps, perf_fns, num_active_somas_post, i_col=1):
        # Plot 1: Performance error
        axs[0].plot(moving_average(perf_errors), color=f"C{i_col}")

        # Plot 2: False positives/negatives
        axs[1].plot(moving_average(perf_fps), color=f"C{i_col}", label="False-positives")
        axs[1].plot(moving_average(perf_fns), linestyle="dashed", color=f"C{i_col}", label="False-negatives")

        # Plot 3: Number of active neurons
        rel_num_active_neurons = moving_average(np.array(num_active_somas_post) / self.p.Network.num_neurons)
        axs[2].plot(rel_num_active_neurons, color=f"C{i_col}")

        return axs

    def plot_bounds(self, axs, error_low, error_up, fp_low, fp_up, fn_low, fn_up, active_somas_low, active_somas_up):
        x = range(self.get_data_size())

        axs[0].fill_between(x, moving_average(error_low), moving_average(error_up), facecolor='grey')

        axs[1].fill_between(x, moving_average(fp_low), moving_average(fp_up), facecolor='grey')
        axs[1].fill_between(x, moving_average(fn_low), moving_average(fn_up), facecolor='grey')

        active_somas_low = moving_average(np.array(active_somas_low) / self.p.Network.num_neurons)
        active_somas_up = moving_average(np.array(active_somas_up) / self.p.Network.num_neurons)
        axs[2].fill_between(x, active_somas_low, active_somas_up, facecolor='grey')

        return axs

    def plot_legend(self, axs):
        axs[0].legend(["Prediction error"])

        axs[1].legend(["False-positives", "False-negatives"])

        axs[2].legend(["Target", "Actual"])

        return axs



class PerformanceSingle(Performance):
    def __init__(self, parameters):
        super().__init__(parameters)

        self.init_data()

    def init_data(self):
        self.data = dict()
        for metric_name in PerformanceMetrics.get_all():
            self.data[metric_name] = [[] for _ in self.p.Experiment.sequences]

    def get_statistic(self, statistic, metric, episode='all', percentile=None):
        if 'all' in episode:
            kwargs = {'axis': 0}
            if statistic == StatisticalMetrics.PERCENTILE and percentile is not None:
                kwargs['q'] = percentile
            result = NP_STATISTICS[statistic](np.array(self.data[metric]), **kwargs)
            if episode == 'all-mean':
                return np.mean(result, **kwargs)
            else:
                return result
        elif episode == 'last':
            return NP_STATISTICS[statistic](np.array(self.data[metric])[:, -1])

    def get_all(self, metric, sequence_id=None):
        if sequence_id is None:
            return self.data[metric]
        else:
            return self.data[metric][sequence_id]

    def get_performance_dict(self, final_result=False, running_avg_perc=0.5):
        performance = dict()
        if final_result:
            for metric_name in PerformanceMetrics.get_all():
                metric = self.get_all(metric_name)
                metric_means = np.mean(metric, axis=0)
                # add final value to dict
                performance[f"{metric_name}_last"] = metric_means[-1]
                # add mean of all training epochs to dict
                performance[f"{metric_name}_running-avg-{running_avg_perc}"] = (
                    np.mean(metric_means[int(len(metric_means)*running_avg_perc):]))
        else:
            performance = self.data
        return performance

    def get_data_size(self):
        size = -1
        for metric_i in self.data.values():
            for seq_i in metric_i:
                if size < 0 or len(seq_i) < size:
                    size = len(seq_i)
        return size

    def add_data_point(self, data_point, metric, sequence_id):
        self.data[metric][sequence_id].append(data_point)

    def plot(self, statistic=StatisticalMetrics.MEAN, sequences="statistic"):
        fig, axs = super().plot(statistic=statistic, sequences=sequences)

        sequence_range = None

        if type(sequences) is str:
            if sequences == "statistic":
                axs = self.plot_seq(axs, self.get_statistic(statistic, PerformanceMetrics.ERROR),
                                    self.get_statistic(statistic, PerformanceMetrics.FP),
                                    self.get_statistic(statistic, PerformanceMetrics.FN),
                                    self.get_statistic(statistic, PerformanceMetrics.ACTIVE_SOMAS), i_col=1)
            elif sequences == "all":
                sequence_range = range(len(self.p.Experiment.sequences))
        elif type(sequences) in [range, list]:
            sequence_range = sequences

        if sequence_range is not None:
            for i_seq in sequence_range:
                self.plot_seq(axs, self.get_all(PerformanceMetrics.ERROR, i_seq),
                              self.get_all(PerformanceMetrics.FP, i_seq),
                              self.get_all(PerformanceMetrics.FN, i_seq),
                              self.get_all(PerformanceMetrics.ACTIVE_SOMAS, i_seq), i_col=i_seq)

        axs = self.plot_legend(axs)

        fig.show()


class PerformanceMulti(Performance):
    def __init__(self, parameters, num_instances):
        super().__init__(parameters)

        self.num_instances = num_instances

        self.init_data()

    def init_data(self):
        self.data = list()
        for i_inst in range(self.num_instances):
            data_inst = dict()
            for metric_name in PerformanceMetrics.get_all():
                data_inst[metric_name] = [[] for _ in self.p.Experiment.sequences]
            self.data.append(data_inst)

    def load_data(self, net, experiment_type, experiment_id, experiment_num):
        folder_path = get_experiment_folder(net, experiment_type, experiment_id, experiment_num)

        for i_inst in range(self.num_instances):
            inst_folder_name = f"{i_inst:02d}"
            inst_folder_path = join(folder_path, inst_folder_name)

            if not os.path.exists(inst_folder_path):
                log.warning(f"Instance folder does not exist: {inst_folder_path}")

            for metric_name in PerformanceMetrics.get_all():
                metric_file_name = f"pf_{metric_name}.npy"
                metric_file_path = join(inst_folder_path, metric_file_name)

                if not os.path.exists(inst_folder_path):
                    log.warning(f"Metric '{metric_name}' for instance {i_inst} could not be loaded because the file "
                                f"doesn't exist: {metric_file_path}")
                    continue

                self.data[i_inst][metric_name] = np.load(metric_file_path).tolist()

    def get_statistic(self, statistic, metric, episode='all', percentile=None):
        if 'all' in episode:
            kwargs = {'axis': 0}
            if statistic == StatisticalMetrics.PERCENTILE and percentile is not None:
                kwargs['q'] = percentile
            mean_inst = [np.mean(np.array(self.data[i_inst][metric]), axis=0) for i_inst in range(self.num_instances)]
            result = NP_STATISTICS[statistic](mean_inst, **kwargs)
            if episode == 'all-mean':
                return np.mean(result)
            else:
                return result
        elif episode == 'last':
            return NP_STATISTICS[statistic]([NP_STATISTICS[statistic](np.array(self.data[metric])[:, -1])
                                             for _ in range(self.num_instances)])

    def get_all(self, metric, sequence_id=None, instance_id=None):
        if instance_id is None:
            log.error("Instance id cannot be non for class PerformanceMulti.")
            return
        if sequence_id is None:
            return self.data[instance_id][metric]
        else:
            return self.data[instance_id][metric][sequence_id]

    def get_performance_dict(self, final_result=False):
        pass

    def get_data_size(self):
        size = -1
        for inst_i in self.data:
            for metric_i in inst_i.values():
                for seq_i in metric_i:
                    if size < 0 or len(seq_i) < size:
                        size = len(seq_i)
        return size

    def add_data_point(self, data_point, metric, sequence_id, instance_id=None):
        self.data[instance_id][metric][sequence_id].append(data_point)

    def plot(self, statistic, sequences=None, instances="statistic"):
        fig, axs = super().plot(statistic=statistic, sequences=sequences)

        axs = self.plot_seq(axs, self.get_statistic(statistic, PerformanceMetrics.ERROR),
                            self.get_statistic(statistic, PerformanceMetrics.FP),
                            self.get_statistic(statistic, PerformanceMetrics.FN),
                            self.get_statistic(statistic, PerformanceMetrics.ACTIVE_SOMAS), i_col=1)
        if statistic == StatisticalMetrics.MEDIAN:
            axs = self.plot_bounds(axs,
                                   error_low=self.get_statistic(StatisticalMetrics.PERCENTILE, PerformanceMetrics.ERROR,
                                                                percentile=5),
                                   error_up=self.get_statistic(StatisticalMetrics.PERCENTILE, PerformanceMetrics.ERROR,
                                                               percentile=95),
                                   fp_low=self.get_statistic(StatisticalMetrics.PERCENTILE, PerformanceMetrics.FP,
                                                             percentile=5),
                                   fp_up=self.get_statistic(StatisticalMetrics.PERCENTILE, PerformanceMetrics.FP,
                                                            percentile=95),
                                   fn_low=self.get_statistic(StatisticalMetrics.PERCENTILE, PerformanceMetrics.FN,
                                                             percentile=5),
                                   fn_up=self.get_statistic(StatisticalMetrics.PERCENTILE, PerformanceMetrics.FN,
                                                            percentile=95),
                                   active_somas_low=self.get_statistic(StatisticalMetrics.PERCENTILE,
                                                                       PerformanceMetrics.ACTIVE_SOMAS,
                                                                       percentile=5),
                                   active_somas_up=self.get_statistic(StatisticalMetrics.PERCENTILE,
                                                                      PerformanceMetrics.ACTIVE_SOMAS,
                                                                      percentile=95)
                                   )

        axs = self.plot_legend(axs)

        fig.show()

        return axs, fig

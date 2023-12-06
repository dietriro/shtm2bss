import time
import numpy as np
import copy
import pickle
import multiprocessing as mp

from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
from pyNN.random import NumpyRNG
from tabulate import tabulate
from abc import ABC, abstractmethod
from quantities import ms

from shtmbss2.common.config import *
from shtmbss2.core.logging import log
from shtmbss2.core.parameters import Parameters
from shtmbss2.core.helpers import (Process, symbol_from_label, NeuronType, RecTypes, id_to_symbol, moving_average,
                                   calculate_trace, psp_max_2_psc_max)
from shtmbss2.common.plot import plot_dendritic_events
from shtmbss2.core.data import (save_config, save_experimental_setup, save_performance_data, save_network_data,
                                save_instance_setup)

if RuntimeConfig.backend == Backends.BRAIN_SCALES_2:
    import pynn_brainscales.brainscales2 as pynn

    from pynn_brainscales.brainscales2.populations import Population, PopulationView
    from pynn_brainscales.brainscales2.connectors import AllToAllConnector, FixedNumberPreConnector
    from pynn_brainscales.brainscales2.standardmodels.cells import SpikeSourceArray
    from pynn_brainscales.brainscales2.standardmodels.synapses import StaticSynapse
    from pynn_brainscales.brainscales2.projections import Projection
elif RuntimeConfig.backend == Backends.NEST:
    import pyNN.nest as pynn

    from pyNN.nest.populations import Population, PopulationView
    from pyNN.nest.connectors import AllToAllConnector, FixedNumberPreConnector
    from pyNN.nest.standardmodels.cells import SpikeSourceArray
    from pyNN.nest.standardmodels.synapses import StaticSynapse
    from pyNN.nest.projections import Projection
else:
    raise Exception(f"Backend {RuntimeConfig.backend} not implemented yet. "
                    f"Please choose among [{Backends.BRAIN_SCALES_2}, {Backends.NEST}]")

ID_PRE = 0
ID_POST = 1
NON_PICKLE_OBJECTS = ["post_somas", "projection", "shtm"]


class SHTMBase(ABC):
    def __init__(self, instance_id=None, seed_offset=None, **kwargs):
        # Load pre-defined parameters
        self.p = Parameters(network_type=self, custom_params=kwargs)
        self.load_params()

        # Declare neuron populations
        self.neurons_exc = None
        self.neurons_inh = None
        self.neurons_ext = None
        self.neurons_add = None

        # Declare connections
        self.ext_to_exc = None
        self.exc_to_exc = None
        self.exc_to_inh = None
        self.inh_to_exc = None

        # Declare recordings
        self.rec_neurons_exc = None
        self.last_ext_spike_time = None
        self.neuron_events = None

        # Declare performance containers
        self.performance_error = None
        self.performance_fp = None
        self.performance_fn = None
        self.performance_active_somas = None
        self.num_active_dendrites_post = None

        self.experiment_num = None
        self.experiment_runs = 0
        self.instance_id = instance_id

        if self.p.Experiment.random_seed:
            if seed_offset is None:
                if self.p.Experiment.seed_offset is None:
                    self.p.Experiment.seed_offset = int(time.time())
            else:
                self.p.Experiment.seed_offset = seed_offset

            instance_offset = self.instance_id if self.instance_id is not None else 0
            np.random.seed(self.p.Experiment.seed_offset + instance_offset)
        else:
            np.random.seed(0)

    def load_params(self, **kwargs):
        self.p = Parameters(network_type=self, custom_params=kwargs)

        self.p.evaluate(recursive=True)

        self.p.Plasticity.tau_h = self.__compute_time_constant_dendritic_rate(dt_stm=self.p.Encoding.dt_stm,
                                                                              dt_seq=self.p.Encoding.dt_seq,
                                                                              target_firing_rate=self.p.Plasticity.y)

        # dynamically calculate new weights, scale by 1/1000 for "original" pynn-nest neurons
        if self.p.Synapses.dyn_weight_calculation:
            self.p.Synapses.w_ext_exc = psp_max_2_psc_max(self.p.Synapses.j_ext_exc_psp,
                                                          self.p.Neurons.Excitatory.tau_m,
                                                          self.p.Neurons.Excitatory.tau_syn_ext,
                                                          self.p.Neurons.Excitatory.c_m) / 1000
            self.p.Synapses.w_exc_inh = psp_max_2_psc_max(self.p.Synapses.j_exc_inh_psp,
                                                          self.p.Neurons.Inhibitory.tau_m,
                                                          self.p.Neurons.Inhibitory.tau_syn_E,
                                                          self.p.Neurons.Inhibitory.c_m)
            self.p.Synapses.w_inh_exc = np.abs(psp_max_2_psc_max(self.p.Synapses.j_inh_exc_psp,
                                                                 self.p.Neurons.Excitatory.tau_m,
                                                                 self.p.Neurons.Excitatory.tau_syn_inh,
                                                                 self.p.Neurons.Excitatory.c_m)) / 1000

    def init_network(self):
        self.init_neurons()
        self.init_connections()
        self.init_external_input()

    def init_neurons(self):
        self.neurons_exc = self.init_all_neurons_exc()

        self.neurons_inh = self.init_neurons_inh()

        self.neurons_ext = Population(self.p.Network.num_symbols, SpikeSourceArray())

    @abstractmethod
    def init_all_neurons_exc(self, num_neurons=None):
        pass

    @abstractmethod
    def init_neurons_exc(self, num_neurons=None):
        pass

    @staticmethod
    def init_neurons_exc_post_preprocess(dendrites, somas):
        for i in range(len(dendrites)):
            dendrites.actual_hwparams[i].multicompartment.enable_conductance = True
            dendrites.actual_hwparams[i].multicompartment.i_bias_nmda = 120
            dendrites.actual_hwparams[i].multicompartment.connect_soma_right = True
            dendrites.actual_hwparams[i].refractory_period.reset_holdoff = 0

        for i in range(len(somas)):
            somas.actual_hwparams[i].multicompartment.connect_soma = True

    @abstractmethod
    def init_neurons_inh(self, num_neurons=None):
        pass

    def init_external_input(self, init_recorder=False, init_performance=False):
        spike_times = [list() for _ in range(self.p.Network.num_symbols)]
        spike_time = None

        sequence_offset = self.p.Encoding.t_exc_start
        for i_seq, sequence in enumerate(self.p.Experiment.sequences):
            for i_element, element in enumerate(sequence):
                spike_time = sequence_offset + i_element * self.p.Encoding.dt_stm
                spike_times[SYMBOLS[element]].append(spike_time)
            sequence_offset = spike_time + self.p.Encoding.dt_seq

        self.last_ext_spike_time = spike_time

        log.debug(f'Spike times:')
        for i_letter, letter_spikes in enumerate(spike_times):
            log.debug(f'{list(SYMBOLS.keys())[i_letter]}: {spike_times[i_letter]}')

        self.neurons_ext.set(spike_times=spike_times)

        if init_performance or self.performance_error is None:
            log.info(f'Initialized external input for sequence(s) {self.p.Experiment.sequences}')
            # Initialize performance containers
            self.performance_error = [[] for _ in self.p.Experiment.sequences]
            self.performance_fp = [[] for _ in self.p.Experiment.sequences]
            self.performance_fn = [[] for _ in self.p.Experiment.sequences]
            self.performance_active_somas = [[] for _ in self.p.Experiment.sequences]
            self.num_active_dendrites_post = [[] for _ in self.p.Experiment.sequences]

    def init_connections(self):
        self.ext_to_exc = []
        for i in range(self.p.Network.num_symbols):
            self.ext_to_exc.append(Projection(
                PopulationView(self.neurons_ext, [i]),
                self.get_neurons(NeuronType.Soma, symbol_id=i),
                AllToAllConnector(),
                synapse_type=StaticSynapse(weight=self.p.Synapses.w_ext_exc, delay=self.p.Synapses.delay_ext_exc),
                receptor_type=self.p.Synapses.receptor_ext_exc))

        self.exc_to_exc = []
        num_connections = int(self.p.Network.num_neurons * self.p.Synapses.p_exc_exc)
        for i in range(self.p.Network.num_symbols):
            for j in range(self.p.Network.num_symbols):
                if i == j:
                    continue
                seed = j + i * self.p.Network.num_symbols
                if self.instance_id is not None:
                    seed += self.instance_id * self.p.Network.num_symbols**2
                if self.p.Experiment.random_seed:
                    seed += self.p.Experiment.seed_offset
                self.exc_to_exc.append(Projection(
                    self.get_neurons(NeuronType.Soma, symbol_id=i),
                    self.get_neurons(NeuronType.Dendrite, symbol_id=j),
                    FixedNumberPreConnector(num_connections, rng=NumpyRNG(seed=j + i * self.p.Network.num_symbols)),
                    synapse_type=StaticSynapse(weight=self.p.Synapses.w_exc_exc, delay=self.p.Synapses.delay_exc_exc),
                    receptor_type=self.p.Synapses.receptor_exc_exc,
                    label=f"exc-exc_{id_to_symbol(i)}>{id_to_symbol(j)}"))

        self.exc_to_inh = []
        for i in range(self.p.Network.num_symbols):
            self.exc_to_inh.append(Projection(
                self.get_neurons(NeuronType.Soma, symbol_id=i),
                PopulationView(self.neurons_inh, [i]),
                AllToAllConnector(),
                synapse_type=StaticSynapse(weight=self.p.Synapses.w_exc_inh, delay=self.p.Synapses.delay_exc_inh),
                receptor_type=self.p.Synapses.receptor_exc_inh))

        self.inh_to_exc = []
        for i in range(self.p.Network.num_symbols):
            self.inh_to_exc.append(Projection(
                PopulationView(self.neurons_inh, [i]),
                self.get_neurons(NeuronType.Soma, symbol_id=i),
                AllToAllConnector(),
                synapse_type=StaticSynapse(weight=self.p.Synapses.w_inh_exc, delay=self.p.Synapses.delay_inh_exc),
                receptor_type=self.p.Synapses.receptor_inh_exc))

    def __compute_time_constant_dendritic_rate(self, dt_stm, dt_seq, target_firing_rate, calibration=0):
        """ Adapted from Bouhadjour et al. 2022
        Compute time constant of the dendritic AP rate,

        The time constant is set such that the rate captures how many dAPs a neuron generated
        all along the period of a batch

        Parameters
        ----------
        calibration : float
        target_firing_rate : float

        Returns
        -------
        float
           time constant of the dendritic AP rate
        """

        t_exc = (((len(self.p.Experiment.sequences[0]) - 1) * dt_stm + dt_seq + calibration)
                 * len(self.p.Experiment.sequences))

        log.debug("\nDuration of a sequence set %d ms" % t_exc)

        return target_firing_rate * t_exc

    def reset(self):
        pass

    @abstractmethod
    def get_neurons(self, neuron_type, symbol_id=None):
        pass

    @abstractmethod
    def get_neuron_data(self, neuron_type, neurons=None, value_type="spikes", symbol_id=None, neuron_id=None,
                        runtime=None, dtype=None):
        pass

    def plot_events(self, neuron_types="all", symbols="all", size=None, x_lim_lower=None, x_lim_upper=None, seq_start=0,
                    seq_end=None, fig_title="", file_path=None, window="initial"):
        if size is None:
            size = (12, 10)

        if type(neuron_types) is str and neuron_types == "all":
            neuron_types = [NeuronType.Dendrite, NeuronType.Soma, NeuronType.Inhibitory]
        elif type(neuron_types) is list:
            pass
        else:
            return

        if window == "initial":
            max_time = self.p.Experiment.runtime
        else:
            max_time = pynn.get_current_time()

        if x_lim_lower is None:
            if window == "initial":
                x_lim_lower = 0.
            else:
                x_lim_lower = pynn.get_current_time() - self.p.Experiment.runtime
        if x_lim_upper is None:
            x_lim_upper = max_time

        if type(symbols) is str and symbols == "all":
            symbols = range(self.p.Network.num_symbols)
        elif type(symbols) is list:
            pass

        if len(symbols) == 1:
            fig, axs = plt.subplots(figsize=size)
        else:
            fig, axs = plt.subplots(self.p.Network.num_symbols, 1, sharex="all", figsize=size)

        if seq_end is None:
            seq_end = seq_start + self.p.Experiment.runtime

        ax = None

        for i_symbol in symbols:
            if len(symbols) == 1:
                ax = axs
            else:
                ax = axs[i_symbol]

            # neurons_all = dict()
            # neurons_all[NeuronType.Dendrite], neurons_all[NeuronType.Soma], = self.neurons_exc[i_symbol]
            # neurons_all[NeuronType.Inhibitory] = pynn.PopulationView(self.neurons_inh, [i_symbol])

            for neurons_i in neuron_types:
                # Retrieve and plot spikes from selected neurons
                spikes = self.get_neuron_data(neuron_type=neurons_i, symbol_id=i_symbol,
                                              value_type=RecTypes.SPIKES, dtype=list)
                if neurons_i == NeuronType.Inhibitory:
                    spikes.append([])
                else:
                    spikes.insert(0, [])
                if neurons_i == NeuronType.Dendrite:
                    spikes_post = self.get_neuron_data(neuron_type=NeuronType.Soma, symbol_id=i_symbol,
                                                       value_type=RecTypes.SPIKES, dtype=list)
                    plot_dendritic_events(ax, spikes[1:], spikes_post, tau_dap=self.p.Neurons.Dendrite.tau_dAP,
                                          color=f"C{neurons_i.ID}", label=neurons_i.NAME.capitalize(),
                                          seq_start=seq_start, seq_end=seq_end)
                else:
                    line_widths = 1.5
                    line_lengths = 1

                    ax.eventplot(spikes, linewidths=line_widths, linelengths=line_lengths,
                                 label=neurons_i.NAME.capitalize(), color=f"C{neurons_i.ID}")

            # Configure the plot layout
            ax.set_xlim(x_lim_lower, x_lim_upper)
            ax.set_ylim(-1, self.p.Network.num_neurons + 1)
            ax.yaxis.set_ticks(range(self.p.Network.num_neurons + 2))
            ax.set_ylabel(id_to_symbol(i_symbol), weight='bold', fontsize=20)
            # ax.grid(True, which='both', axis='both')

            # Generate y-tick-labels based on number of neurons per symbol
            y_tick_labels = ['Inh', '', '0'] + ['' for _ in range(self.p.Network.num_neurons - 2)] + [
                str(self.p.Network.num_neurons - 1)]
            ax.set_yticklabels(y_tick_labels, rotation=45, fontsize=18)

        # Create custom legend for all plots
        custom_lines = [Line2D([0], [0], color=f"C{n.ID}", label=n.NAME.capitalize(), lw=3) for n in neuron_types]

        ax.set_xlabel("Time [ms]", fontsize=26, labelpad=14)
        ax.xaxis.set_ticks(np.arange(x_lim_lower, x_lim_upper, self.p.Encoding.dt_stm / 2))
        ax.tick_params(axis='x', labelsize=18)

        plt.figlegend(handles=custom_lines, loc=(0.377, 0.885), ncol=3, labelspacing=0., fontsize=18, fancybox=True,
                      borderaxespad=4)

        fig.text(0.01, 0.5, "Symbol & Neuron ID", va="center", rotation="vertical", fontsize=26)

        fig.suptitle(fig_title, x=0.5, y=0.99, fontsize=26)
        fig.show()

        if file_path is not None:
            plt.savefig(f"{file_path}.pdf")

            pickle.dump(fig, open(f'{file_path}.fig.pickle',
                                  'wb'))  # This is for Python 3 - py2 may need `file` instead of `open`

    def plot_v_exc(self, alphabet_range, neuron_range='all', size=None, neuron_type=NeuronType.Soma, runtime=0.1,
                   show_legend=False, file_path=None):
        if size is None:
            size = (12, 10)

        if type(neuron_range) is str and neuron_range == 'all':
            neuron_range = range(self.p.Network.num_neurons)
        elif type(neuron_range) is list or type(neuron_range) is range:
            pass
        else:
            return

        spike_times = [[]]
        header_spikes = list()

        fig, ax = plt.subplots(figsize=size)

        for alphabet_id in alphabet_range:
            # retrieve and save spike times
            spikes = self.get_neuron_data(neuron_type, value_type=RecTypes.SPIKES,
                                          symbol_id=alphabet_id, dtype=np.ndarray)
            for neuron_id in neuron_range:
                # add spikes to list for printing
                spike_times[0].append(spikes[:, 1].round(5).tolist())
                header_spikes.append(f"{id_to_symbol(alphabet_id)}[{neuron_id}]")

                # retrieve voltage data
                data_v = self.get_neuron_data(neuron_type, value_type=RecTypes.V, symbol_id=alphabet_id,
                                              neuron_id=neuron_id)

                ax.plot(data_v.times, data_v, alpha=0.5, label=header_spikes[-1])

        # ax.xaxis.set_ticks(np.arange(0.02, 0.06, 0.01))
        ax.tick_params(axis='x', labelsize=18)
        ax.tick_params(axis='y', labelsize=18)

        ax.set_xlabel("Time [ms]", labelpad=14, fontsize=26)
        ax.set_ylabel("Membrane Voltage [a.u.]", labelpad=14, fontsize=26)

        if show_legend:
            plt.legend()

        # Print spike times
        print(tabulate(spike_times, headers=header_spikes) + '\n')

        fig.show()

        if file_path is not None:
            plt.savefig(f"{file_path}.pdf")

            pickle.dump(fig, open(f'{file_path}.fig.pickle',
                                  'wb'))  # This is for Python 3 - py2 may need `file` instead of `open`

    def plot_performance(self, sequences="mean"):
        fig, axs = plt.subplots(1, 3, figsize=[12, 5])

        sequence_range = None

        if type(sequences) is str:
            if sequences == "mean":
                axs = self.__plot_performance_seq(axs, np.mean(self.performance_error, axis=0),
                                                  np.mean(self.performance_fp, axis=0),
                                                  np.mean(self.performance_fn, axis=0),
                                                  np.mean(self.performance_active_somas, axis=0), i_col=1)
            elif sequences == "all":
                sequence_range = range(len(self.p.Experiment.sequences))
        elif type(sequences) in [range, list]:
            sequence_range = sequences

        if sequence_range is not None:
            for i_seq in sequence_range:
                self.__plot_performance_seq(axs, self.performance_error[i_seq],
                                            self.performance_fp[i_seq],
                                            self.performance_fn[i_seq],
                                            self.performance_active_somas[i_seq], i_col=i_seq)

        axs[0].set_ylabel("Prediction error")
        axs[0].set_xlabel("# Training Episodes")

        axs[1].set_ylabel("Rel. frequency")
        axs[1].set_xlabel("# Training Episodes")
        axs[1].legend(["False-positives", "False-negatives"])

        target_pattern_size_line = [self.p.Network.pattern_size / self.p.Network.num_neurons
                                    for _ in self.performance_active_somas[0]]
        axs[2].plot(target_pattern_size_line, linestyle="dashed", color=f"grey",
                    label=f"Target pattern size ({self.p.Network.pattern_size})")
        axs[2].set_ylabel("Rel. no. of active neurons")
        axs[2].set_xlabel("# Training Episodes")

        fig.tight_layout()
        fig.show()

    def __plot_performance_seq(self, axs, perf_errors, perf_fps, perf_fns, num_active_somas_post, i_col=1):
        # Plot 1: Performance error
        axs[0].plot(moving_average(perf_errors), color=f"C{i_col}")

        # Plot 2: False positives/negatives
        axs[1].plot(moving_average(perf_fps), color=f"C{i_col}", label="False-positives")
        axs[1].plot(moving_average(perf_fns), linestyle="dashed", color=f"C{i_col}", label="False-negatives")

        # Plot 3: Number of active neurons
        rel_num_active_neurons = moving_average(np.array(num_active_somas_post) / self.p.Network.num_neurons)
        axs[2].plot(rel_num_active_neurons, color=f"C{i_col}")

        return axs

    def compute_prediction_performance(self, method=PerformanceType.ALL_SYMBOLS):
        log.info(f"Computing performance for {len(self.p.Experiment.sequences)} Sequences.")

        ratio_fp_activation = 0.5
        ratio_fn_activation = 0.5

        t_min = self.p.Encoding.t_exc_start

        for i_seq, seq in enumerate(self.p.Experiment.sequences):
            seq_error = list()
            seq_fp = list()
            seq_fn = list()
            seq_num_active_somas_post = list()
            seq_num_active_dendrites_post = list()

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
                                                 self.neuron_events[NeuronType.Dendrite][i_symbol] for item in sublist])

                    # get somatic spikes per subpopulation
                    num_som_spikes[i_symbol] = np.sum([t_min_next < item < t_max_next for sublist in
                                                       self.neuron_events[NeuronType.Soma][i_symbol] for item in
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

                seq_error.append(error)
                seq_fp.append(false_positive)
                seq_fn.append(false_negative)
                seq_num_active_somas_post.append(num_som_spikes[SYMBOLS[element]])
                seq_num_active_dendrites_post.append(num_dAPs[SYMBOLS[element]])

            self.performance_error[i_seq].append(np.mean(seq_error))
            self.performance_fp[i_seq].append(np.mean(seq_fp))
            self.performance_fn[i_seq].append(np.mean(seq_fn))
            self.performance_active_somas[i_seq].append(np.mean(seq_num_active_somas_post))
            self.num_active_dendrites_post[i_seq].append(np.mean(seq_num_active_dendrites_post))

    def get_performance_dict(self, final_result=False):
        performance = dict()
        if final_result:
            for metric_name in ["performance_error", "performance_fp", "performance_fn", "performance_active_somas"]:
                metric = getattr(self, metric_name)
                metric_means = np.mean(metric, axis=0)
                # add final value to dict
                performance[metric_name] = metric_means[-1]
                # add mean of all training epochs to dict
                performance[metric_name] = np.mean(metric_means)
        else:
            performance['performance_error'] = self.performance_error
            performance['performance_fp'] = self.performance_fp
            performance['performance_fn'] = self.performance_fn
            performance['performance_active_somas'] = self.performance_active_somas
            performance['num_active_dendrites_post'] = self.num_active_dendrites_post
        return performance


    def save_full_state(self):
        log.debug("Saving full state of network and experiment.")

        if self.p.Experiment.type == ExperimentType.EVAL_MULTI:
            if self.instance_id is not None and self.instance_id == 0:
                self.experiment_num = save_experimental_setup(net=self, experiment_num=self.experiment_num,
                                                              instance_id=self.instance_id)
            save_instance_setup(net=self, performance=self.get_performance_dict(final_result=True),
                                experiment_num=self.experiment_num, instance_id=self.instance_id)
        else:
            self.experiment_num = save_experimental_setup(net=self, experiment_num=self.experiment_num,
                                                          instance_id=self.instance_id)

        if ((self.p.Experiment.type == ExperimentType.EVAL_MULTI and
                self.instance_id is not None and self.instance_id == 1) or
                self.p.Experiment.type == ExperimentType.EVAL_SINGLE):
            self.experiment_num = save_experimental_setup(net=self, experiment_num=self.experiment_num,
                                                          instance_id=self.instance_id)
        save_config(net=self, instance_id=self.instance_id)
        data = [self.performance_error, self.performance_fp, self.performance_fn, self.performance_active_somas,
                self.num_active_dendrites_post]
        metric_names = ["pf_error", "pf_fps", "pf_fns", "pf_active_somas", "pf_active_dend"]
        save_performance_data(data, metric_names, self, self.experiment_num, instance_id=self.instance_id)
        save_network_data(self, self.experiment_num, instance_id=self.instance_id)

    def __str__(self):
        return type(self).__name__


class SHTMTotal(SHTMBase, ABC):
    def __init__(self, log_permanence=None, log_weights=None, w_exc_inh_dyn=None, plasticity_cls=None, instance_id=None,
                 seed_offset=None, **kwargs):
        super().__init__(instance_id=instance_id, seed_offset=seed_offset, **kwargs)

        self.con_plastic = None
        self.w_exc_inh_dyn = w_exc_inh_dyn
        self.trace_dendrites = self.trace_dendrites = np.zeros(shape=(self.p.Network.num_symbols,
                                                                      self.p.Network.num_neurons))

        if plasticity_cls is None:
            self.plasticity_cls = Plasticity
        else:
            self.plasticity_cls = plasticity_cls

        if log_permanence is None:
            self.log_permanence = list()
        elif type(log_permanence) is str and log_permanence.lower() == "all":
            self.log_permanence = range(self.p.Network.num_symbols ** 2 - self.p.Network.num_symbols)
        else:
            self.log_permanence = log_permanence

        if log_weights is None:
            self.log_weights = list()
        elif type(log_weights) is str and log_weights.lower() == "all":
            self.log_weights = range(self.p.Network.num_symbols ** 2 - self.p.Network.num_symbols)
        else:
            self.log_weights = log_weights

    def init_connections(self, debug=False):
        super().init_connections()

        self.con_plastic = list()

        for i_plastic in range(len(self.exc_to_exc)):
            # Retrieve id (letter) of post synaptic neuron population
            symbol_post = self.exc_to_exc[i_plastic].label.split('_')[1].split('>')[1]
            # Create population view of all post synaptic somas
            post_somas = PopulationView(self.get_neurons(NeuronType.Soma, symbol_id=SYMBOLS[symbol_post]),
                                        list(range(self.p.Network.num_neurons)))
            if self.p.Synapses.dyn_inh_weights:
                proj_post_soma_inh = self.exc_to_inh[SYMBOLS[symbol_post]]
            else:
                proj_post_soma_inh = None

            self.con_plastic.append(self.plasticity_cls(self.exc_to_exc[i_plastic], post_somas=post_somas, shtm=self,
                                                        proj_post_soma_inh=proj_post_soma_inh, index=i_plastic,
                                                        debug=debug, **self.p.Plasticity.dict()))

        for i_perm in self.log_permanence:
            self.con_plastic[i_perm].enable_permanence_logging()
        # ToDo: Find out why accessing the weights before first simulation causes switch in connections between symbols.
        # This seems to be a nest issue.
        # for i_perm in self.log_weights:
        #     self.con_plastic[i_perm].enable_weights_logging()

    def print_permanence_diff(self):
        for i_perm in self.log_permanence:
            permanence_diff = self.con_plastic[i_perm].permanences[-1] - self.con_plastic[i_perm].permanences[0]
            print(
                f"Permanence diff for {self.con_plastic[i_perm].projection.label} ({i_perm}): {list(permanence_diff)}")

    def plot_permanence_diff(self):
        fig, axs = plt.subplots(len(self.log_permanence), 1, sharex="all", figsize=(22, 7))

        all_connection_ids = set()
        for i_perm in self.log_permanence:
            connection_ids = self.con_plastic[i_perm].get_all_connection_ids()
            all_connection_ids.update(connection_ids)
        all_connection_ids = sorted(all_connection_ids)

        for i_perm in self.log_permanence:
            permanence_diff = self.con_plastic[i_perm].permanences[-1] - self.con_plastic[i_perm].permanences[0]
            connection_ids = self.con_plastic[i_perm].get_all_connection_ids()

            all_permanence_diff = np.zeros(len(all_connection_ids))
            for i_all_cons, con_ids in enumerate(all_connection_ids):
                if con_ids in connection_ids:
                    all_permanence_diff[i_all_cons] = permanence_diff[connection_ids.index(con_ids)]

            colors = ['C0' if p >= 0 else 'C1' for p in all_permanence_diff]

            axs[i_perm].bar(range(len(all_connection_ids)), all_permanence_diff, color=colors)
            axs[i_perm].set_ylabel(self.con_plastic[i_perm].projection.label.split('_')[1], weight='bold')
            y_min = round(min(all_permanence_diff), 1)
            y_max = round(max(all_permanence_diff), 1)
            if y_min == y_max == 0:
                axs[i_perm].yaxis.set_ticks([0.0])
            else:
                axs[i_perm].yaxis.set_ticks([y_min, y_max])
            axs[i_perm].xaxis.set_ticks(range(len(all_connection_ids)), all_connection_ids, rotation=45)
            axs[i_perm].grid(True, which='both', axis='both')

        axs[-1].set_xlabel("Connection [#]")
        fig.tight_layout(pad=1.0)
        fig.text(0.02, 0.5, "Permanence diff / connection direction", va="center", rotation="vertical")

    def plot_permanence_history(self, plot_con_ids='all'):
        if type(plot_con_ids) is str and plot_con_ids == 'all':
            plot_con_ids = self.log_permanence
        elif type(plot_con_ids) is list:
            pass
        else:
            return

        fig, axs = plt.subplots(len(plot_con_ids), 1, sharex="all", figsize=(14, len(plot_con_ids) * 4))

        for i_plot, i_perm in enumerate(plot_con_ids):
            permanence = self.con_plastic[i_perm].permanences
            num_connections = len(permanence)

            # Plot all previous permanences as a line over time
            axs[i_plot].plot(range(num_connections), permanence)

            axs[i_plot].set_ylabel(self.con_plastic[i_perm].projection.label.split('_')[1], weight='bold')
            axs[i_plot].grid(True, which='both', axis='both')

        axs[-1].set_xlabel("Number of learning phases")
        fig.text(0.02, 0.5, "Permanence / connection direction", va="center", rotation="vertical")

        fig.show()

    def plot_weight_diff(self):
        fig, axs = plt.subplots(len(self.log_weights), 1, sharex="all", figsize=(10, 7))

        for i_perm in self.log_weights:
            weights_diff = self.con_plastic[i_perm].weights[-1] - self.con_plastic[i_perm].weights[0]
            num_connections = len(weights_diff)

            colors = ['C0' if p >= 0 else 'C1' for p in weights_diff]

            axs[i_perm].bar(range(num_connections), weights_diff, color=colors)
            axs[i_perm].set_ylabel(self.con_plastic[i_perm].projection.label.split('_')[1], weight='bold')
            axs[i_perm].yaxis.set_ticks([0])
            axs[i_perm].xaxis.set_ticks(range(0, num_connections))
            axs[i_perm].grid(True, which='both', axis='both')

        axs[-1].set_xlabel("Connection [#]")
        fig.text(0.02, 0.5, "Weights diff / connection direction", va="center", rotation="vertical")

    def __retrieve_neuron_data(self):
        self.neuron_events = dict()

        for neuron_type in NeuronType.get_all_types():
            self.neuron_events[neuron_type] = list()
            for i_symbol in range(self.p.Network.num_symbols):
                events = self.get_neuron_data(neuron_type, value_type=RecTypes.SPIKES, symbol_id=i_symbol,
                                              dtype=list)
                self.neuron_events[neuron_type].append(events)

    def get_spike_times(self, runtime, dt):
        log.detail("Calculating spike times")

        times = np.linspace(0., runtime, int(runtime / dt))

        spike_times_dendrite = np.zeros((self.p.Network.num_symbols, self.p.Network.num_neurons, len(times)),
                                        dtype=np.int8)
        spike_times_soma = np.zeros((self.p.Network.num_symbols, self.p.Network.num_neurons, len(times)), dtype=np.int8)

        for i_symbol in range(self.p.Network.num_symbols):
            for i_dendrite, dendrite_spikes in enumerate(self.get_neuron_data(NeuronType.Dendrite, symbol_id=i_symbol,
                                                                              value_type=RecTypes.SPIKES, dtype=list)):
                for spike_time in dendrite_spikes:
                    spike_id = int(spike_time / times[1])
                    spike_times_dendrite[i_symbol, i_dendrite, spike_id] = 1

            for i_soma, soma_spikes in enumerate(self.get_neuron_data(NeuronType.Soma, symbol_id=i_symbol,
                                                                      value_type=RecTypes.SPIKES)):
                for spike_time in soma_spikes:
                    spike_id = int(spike_time / times[1])
                    spike_times_soma[i_symbol, i_soma, spike_id] = 1

        return spike_times_dendrite, spike_times_soma

    def __update_dendritic_trace(self):
        for i_symbol in range(self.p.Network.num_symbols):
            for i_neuron in range(self.p.Network.num_neurons):
                events = np.array(self.neuron_events[NeuronType.Dendrite][i_symbol][i_neuron])
                self.trace_dendrites[i_symbol, i_neuron] = calculate_trace(self.trace_dendrites[i_symbol, i_neuron],
                                                                           0, self.p.Experiment.runtime,
                                                                           events, self.p.Plasticity.tau_h)

    def set_weights_exc_exc(self, new_weight, con_id, post_ids=None, p_con=1.0):
        # ToDo: Find out why this is not working in Nest after one simulation, only before all simulations
        weights = self.con_plastic[con_id].projection.get("weight", format="array")

        if post_ids is None:
            post_ids = range(weights.shape[1])

        for i in post_ids:
            pre_ids = np.logical_not(np.isnan(weights[:, i]))
            pre_ids = pre_ids[:int(p_con * len(pre_ids))]
            weights[pre_ids, i] = new_weight

        self.con_plastic[con_id].projection.set(weight=weights)
        self.con_plastic[con_id].w_mature = new_weight

        return self.con_plastic[con_id].projection.get("weight", format="array")

    def run(self, runtime=None, steps=None, plasticity_enabled=True, dyn_exc_inh=False, run_type=RunType.SINGLE):
        if runtime is None:
            runtime = self.p.Experiment.runtime
        if steps is None:
            steps = self.p.Experiment.episodes

        if type(runtime) is str:
            if str(runtime).lower() == 'max':
                runtime = self.last_ext_spike_time + (self.p.Encoding.dt_seq - self.p.Encoding.t_exc_start)
        elif type(runtime) is float or type(runtime) is int:
            pass
        elif runtime is None:
            log.debug("No runtime specified. Setting runtime to last spike time + 2xdt_stm")
            runtime = self.last_ext_spike_time + (self.p.Encoding.dt_seq - self.p.Encoding.t_exc_start)
        else:
            log.error("Error! Wrong runtime")

        self.p.Experiment.runtime = runtime

        for t in range(steps):
            log.info(f'Running emulation step {t + 1}/{steps}')

            # reset the simulator and the network state if not first run
            if pynn.get_current_time() > 0 and t > 0:
                self.reset()

            # set start time to 0.0 because
            # - nest is reset and always starts with 0.0
            # - bss2 resets the time itself after each run to 0.0
            sim_start_time = 0.0
            log.detail(f"Current time: {sim_start_time}")

            pynn.run(runtime)

            self.__retrieve_neuron_data()

            if self.p.Performance.compute_performance:
                self.compute_prediction_performance(method=self.p.Performance.method)

            if plasticity_enabled:
                if run_type == RunType.MULTI:
                    log.warn(
                        f"Multi-core version of plasticity calculation is currently not working. Please choose the "
                        f"single-core version. Not calculating plasticity.")
                    # self.__run_plasticity_parallel(runtime, sim_start_time, dyn_exc_inh=dyn_exc_inh)
                elif run_type == RunType.SINGLE:
                    self.__run_plasticity_singular(runtime, sim_start_time, dyn_exc_inh=dyn_exc_inh)

            if self.p.Experiment.autosave and self.p.Experiment.autosave_epoches > 0:
                if (t + 1) % self.p.Experiment.autosave_epoches == 0:
                    self.p.Experiment.episodes = self.experiment_runs + t + 1
                    self.save_full_state()

        self.experiment_runs += steps

    def __run_plasticity_singular(self, runtime, sim_start_time, dyn_exc_inh=False):
        log.info("Starting plasticity calculations")

        active_synapse_post = np.zeros((self.p.Network.num_symbols, self.p.Network.num_neurons))

        # Calculate plasticity for each synapse
        for i_plasticity, plasticity in enumerate(self.con_plastic):
            plasticity(runtime, sim_start_time=sim_start_time)
            log.debug(f"Finished plasticity calculation {i_plasticity + 1}/{len(self.con_plastic)}")

            if dyn_exc_inh:
                w = self.exc_to_exc[i_plasticity].get("weight", format="array")
                letter_id = SYMBOLS[plasticity.get_post_symbol()]
                active_synapse_post[letter_id, :] = np.logical_or(active_synapse_post[letter_id, :],
                                                                  np.any(w > 0, axis=0))

        if dyn_exc_inh and self.w_exc_inh_dyn is not None:
            for i_inh in range(self.p.Network.num_symbols):
                w = self.exc_to_inh.get("weight", format="array")
                w[active_synapse_post[i_inh, :]] = self.w_exc_inh_dyn

        self.__update_dendritic_trace()

    def __run_plasticity_parallel(self, runtime, sim_start_time, dyn_exc_inh=False):
        log.info("Starting plasticity calculations")

        active_synapse_post = np.zeros((self.p.Network.num_symbols, self.p.Network.num_neurons))

        q_plasticity = mp.Queue()

        # Calculate plasticity for each synapse
        processes = []
        for i_plasticity, plasticity in enumerate(self.con_plastic):
            log.debug(f'Starting plasticity calculation for {i_plasticity}')
            processes.append(Process(target=plasticity, args=(plasticity, runtime, sim_start_time, q_plasticity)))
            processes[i_plasticity].start()

        num_finished_plasticities = 0
        while num_finished_plasticities < len(self.con_plastic):
            log.debug(f'Waiting for plasticity calculation [{num_finished_plasticities + 1}/{len(self.con_plastic)}]')
            con_plastic = q_plasticity.get()
            self.__update_con_plastic(con_plastic)
            num_finished_plasticities += 1

        for i_plasticity, plasticity in enumerate(self.con_plastic):
            processes[i_plasticity].join()

            log.debug(f"Finished plasticity calculation {i_plasticity + 1}/{len(self.con_plastic)}")

            # Check if an exception occurred in the sub-process, then raise this exception
            if processes[i_plasticity].exception:
                exc, trc = processes[i_plasticity].exception
                print(trc)
                raise exc

            if dyn_exc_inh:
                w = self.exc_to_exc[i_plasticity].get("weight", format="array")
                letter_id = SYMBOLS[plasticity.get_post_symbol()]
                active_synapse_post[letter_id, :] = np.logical_or(active_synapse_post[letter_id, :],
                                                                  np.any(w > 0, axis=0))

        if dyn_exc_inh and self.w_exc_inh_dyn is not None:
            for i_inh in range(self.p.Network.num_symbols):
                w = self.exc_to_inh.get("weight", format="array")
                w[active_synapse_post[i_inh, :]] = self.w_exc_inh_dyn

        self.__update_dendritic_trace()

    def __update_con_plastic(self, new_con_plastic):
        for obj_name in NON_PICKLE_OBJECTS:
            setattr(new_con_plastic, obj_name, getattr(self.con_plastic[new_con_plastic.id], obj_name))

        for obj_name, obj_value in new_con_plastic.__dict__.items():
            if not (obj_name.startswith('_') or callable(obj_value)):
                if obj_name == "proj_post_soma_inh":
                    if self.con_plastic[new_con_plastic.id].proj_post_soma_inh is not None:
                        self.con_plastic[new_con_plastic.id].proj_post_soma_inh.set(
                            weight=new_con_plastic.get("weight", format="array"))
                elif obj_name == "projection_weight":
                    if new_con_plastic.projection_weight is not None:
                        self.con_plastic[new_con_plastic.id].projection.set(weight=new_con_plastic.projection_weight)
                elif obj_name not in NON_PICKLE_OBJECTS:
                    setattr(self.con_plastic[new_con_plastic.id], obj_name, getattr(new_con_plastic, obj_name))


class Plasticity(ABC):
    def __init__(self, projection: Projection, post_somas, shtm, index, proj_post_soma_inh=None, debug=False,
                 learning_factor=None, permanence_init_min=None, permanence_init_max=None, permanence_max=None,
                 threshold=None, w_mature=None, y=None, lambda_plus=None,
                 lambda_minus=None, lambda_h=None, target_rate_h=None, tau_plus=None, tau_h=None, delta_t_min=None,
                 delta_t_max=None, dt=None, **kwargs):
        # custom objects
        self.projection = projection
        self.proj_post_soma_inh = proj_post_soma_inh
        self.shtm: SHTMTotal = shtm
        self.post_somas = post_somas

        # editable/changing variables
        self.permanence_min = np.asarray(np.random.randint(permanence_init_min, permanence_init_max,
                                                           size=(len(self.projection),)), dtype=float)
        self.permanence = copy.copy(self.permanence_min)
        self.permanences = None
        self.weights = None
        self.x = np.zeros((len(self.projection.pre)))
        self.z = np.zeros((len(self.projection.post)))

        self.debug = debug
        self.id = index
        self.projection_weight = None

        # parameters - loaded from file
        self.permanence_max = permanence_max
        self.w_mature = w_mature
        self.tau_plus = tau_plus
        self.tau_h = tau_h
        self.target_rate_h = target_rate_h
        self.y = y
        self.delta_t_min = delta_t_min
        self.delta_t_max = delta_t_max
        self.dt = dt
        self.threshold = np.ones((len(self.projection))) * threshold
        self.lambda_plus = lambda_plus * learning_factor
        self.lambda_minus = lambda_minus * learning_factor
        self.lambda_h = lambda_h * learning_factor

        self.symbol_id_pre = SYMBOLS[symbol_from_label(self.projection.label, ID_PRE)]
        self.symbol_id_post = SYMBOLS[symbol_from_label(self.projection.label, ID_POST)]

    def rule(self, permanence, threshold, x, z, runtime, permanence_min,
             neuron_spikes_pre, neuron_spikes_post_soma, neuron_spikes_post_dendrite,
             delay, sim_start_time=0.0):
        last_spike_pre = 0

        neuron_spikes_pre = np.array(neuron_spikes_pre)
        neuron_spikes_post_soma = np.array(neuron_spikes_post_soma)
        neuron_spikes_post_dendrite = np.array(neuron_spikes_post_dendrite)

        log.debug(f"{self.id}  permanence before: {permanence}")

        # loop through pre-synaptic spikes
        for spike_pre in neuron_spikes_pre:
            # calculate temorary x/z value (pre-synaptic/post-dendritic decay)
            x = x * np.exp(-(spike_pre - last_spike_pre) / self.tau_plus) + 1.0

            # loop through post-synaptic spikes between pre-synaptic spikes
            for spike_post in neuron_spikes_post_soma:
                spike_dt = (spike_post + delay) - spike_pre

                # check if spike-dif is in boundaries
                if self.delta_t_min < spike_dt < self.delta_t_max:
                    # calculate temorary x value (pre synaptic decay)
                    x_tmp = x * np.exp(-spike_dt / self.tau_plus)
                    z_tmp = calculate_trace(z, sim_start_time, spike_post, neuron_spikes_post_dendrite,
                                            self.tau_h)

                    # hebbian learning
                    permanence = self.__facilitate(permanence, x_tmp)
                    log.debug(f"{self.id}  permanence facilitate: {permanence}")

                    permanence = self.__homeostasis_control(permanence, z_tmp, permanence_min)
                    log.debug(f"{self.id}  permanence homeostasis: {permanence}")

            permanence = self.__depress(permanence, permanence_min)
            last_spike_pre = spike_pre
            log.debug(f"{self.id}  permanence depression: {permanence}")

        log.debug(f"{self.id}  permanence after: {permanence}")

        # update x (kplus) and z
        x = x * np.exp(-(runtime - last_spike_pre) / self.tau_plus)

        if permanence >= threshold:
            mature = True
        else:
            mature = False

        return permanence, x, mature

    def __facilitate(self, permanence, x):
        mu = 0
        clip = np.power(1.0 - (permanence / self.permanence_max), mu)
        permanence_norm = (permanence / self.permanence_max) + (self.lambda_plus * x * clip)
        return min(permanence_norm * self.permanence_max, self.permanence_max)

    def __homeostasis_control(self, permanence, z, permanence_min):
        permanence = permanence + self.lambda_h * (self.target_rate_h - z) * self.permanence_max
        return max(min(permanence, self.permanence_max), permanence_min)

    def __depress(self, permanence, permanence_min):
        permanence = permanence - self.lambda_minus * self.permanence_max
        return max(permanence_min, permanence)

    def enable_permanence_logging(self):
        self.permanences = [np.copy(self.permanence)]

    def enable_weights_logging(self):
        self.weights = [np.copy(self.projection.get("weight", format="array").flatten())]

    def get_pre_symbol(self):
        return symbol_from_label(self.projection.label, ID_PRE)

    def get_post_symbol(self):
        return symbol_from_label(self.projection.label, ID_POST)

    def get_connection_ids(self, connection_id):
        connection_ids = (f"{self.get_connection_id_pre(self.get_connections()[connection_id])}>"
                          f"{self.get_connection_id_post(self.get_connections()[connection_id])}")
        return connection_ids

    @abstractmethod
    def get_connection_id_pre(self, connection):
        pass

    @abstractmethod
    def get_connection_id_post(self, connection):
        pass

    def get_all_connection_ids(self):
        connection_ids = []
        for con in self.get_connections():
            connection_ids.append(f"{self.get_connection_id_pre(con)}>{self.get_connection_id_post(con)}")
        return connection_ids

    @abstractmethod
    def get_connections(self):
        pass

    def __call__(self, runtime: float, sim_start_time=0.0, q_plasticity=None):
        spikes_pre = self.shtm.neuron_events[NeuronType.Soma][self.symbol_id_pre]
        spikes_post_dendrite = self.shtm.neuron_events[NeuronType.Dendrite][self.symbol_id_post]
        spikes_post_soma = self.shtm.neuron_events[NeuronType.Soma][self.symbol_id_post]

        weight = self.projection.get("weight", format="array")

        for c, connection in enumerate(self.get_connections()):
            i = self.get_connection_id_post(connection)
            j = self.get_connection_id_pre(connection)
            neuron_spikes_pre = spikes_pre[j]
            neuron_spikes_post_dendrite = spikes_post_dendrite[i]
            neuron_spikes_post_soma = spikes_post_soma[i]
            z = self.shtm.trace_dendrites[self.symbol_id_post, i]

            if self.debug:
                log.debug(f"Permanence calculation for connection {c} [{i}, {j}]")
                log.debug(f"Spikes pre [soma]: {neuron_spikes_pre}")
                log.debug(f"Spikes post [dend]: {neuron_spikes_post_dendrite}")
                log.debug(f"Spikes post [soma]: {neuron_spikes_post_soma}")

            permanence, x, mature = self.rule(permanence=self.permanence[c],
                                              threshold=self.threshold[c],
                                              runtime=runtime, x=self.x[j], z=z,
                                              permanence_min=self.permanence_min[c],
                                              neuron_spikes_pre=neuron_spikes_pre,
                                              neuron_spikes_post_soma=neuron_spikes_post_soma,
                                              neuron_spikes_post_dendrite=neuron_spikes_post_dendrite,
                                              delay=self.shtm.p.Synapses.delay_exc_exc,
                                              sim_start_time=sim_start_time)

            self.permanence[c] = permanence
            self.x[j] = x

            if mature:
                weight[j, i] = self.w_mature
                if self.proj_post_soma_inh is not None:
                    weight_inh = self.proj_post_soma_inh.get("weight", format="array")
                    weight_inh[i, :] = 250
                    log.debug(f"+ | W_inh[{i}] = {weight_inh.flatten()}")
                    self.proj_post_soma_inh.set(weight=weight_inh)
            else:
                weight[j, i] = 0
                if self.proj_post_soma_inh is not None:
                    weight_inh = self.proj_post_soma_inh.get("weight", format="array")
                    weight_inh_old = np.copy(weight_inh)
                    weight_inh[i, :] = 0
                    if np.sum(weight_inh_old.flatten() - weight_inh.flatten()) == 0:
                        log.debug(f"- | W_inh[{i}] = {weight_inh.flatten()}")
                    self.proj_post_soma_inh.set(weight=weight_inh)

        self.projection.set(weight=weight)

        if self.permanences is not None:
            self.permanences.append(np.copy(np.round(self.permanence, 6)))
        if self.weights is not None:
            self.weights.append(
                np.copy(np.round(self.projection.get("weight", format="array").flatten(), 6)))

        log.debug(f'Finished execution of plasticity for {self.id}')

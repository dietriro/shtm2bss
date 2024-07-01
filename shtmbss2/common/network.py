import time
import numpy as np
import copy
import pickle
import yaml
import sys
import multiprocessing as mp

from types import ModuleType, FunctionType
from gc import get_referents
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
from pyNN.random import NumpyRNG
from tabulate import tabulate
from abc import ABC, abstractmethod
from copy import deepcopy

from shtmbss2.common.config import *
from shtmbss2.core.logging import log
from shtmbss2.core.parameters import NetworkParameters, PlottingParameters
from shtmbss2.core.performance import PerformanceSingle
from shtmbss2.core.helpers import (Process, symbol_from_label, id_to_symbol, calculate_trace,
                                   psp_max_2_psc_max)
from shtmbss2.common.config import NeuronType, RecTypes
from shtmbss2.common.plot import plot_dendritic_events
from shtmbss2.core.data import (save_experimental_setup, save_instance_setup, get_experiment_folder)

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


# Custom objects know their class.
# Function objects seem to know way too much, including modules.
# Exclude modules as well.
BLACKLIST = type, ModuleType, FunctionType


class SHTMBase(ABC):
    def __init__(self, experiment_type=ExperimentType.EVAL_SINGLE, experiment_id=None, experiment_num=None,
                 experiment_subnum=None, instance_id=None, seed_offset=None, p=None, **kwargs):
        if experiment_type == ExperimentType.OPT_GRID:
            self.optimized_parameters = kwargs
        else:
            self.optimized_parameters = None

        # Load pre-defined parameters
        self.p_plot: PlottingParameters = PlottingParameters(network_type=self)
        if p is None:
            self.p: NetworkParameters = NetworkParameters(network_type=self)
            self.load_params(experiment_type, experiment_id, experiment_num, instance_id, **kwargs)
        else:
            self.p: NetworkParameters = deepcopy(p)
        self.p.experiment.type = experiment_type

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
        self.spike_times_ext = None
        self.last_ext_spike_time = None
        self.neuron_events = None

        self.experiment_id = experiment_id
        self.experiment_num = experiment_num
        self.experiment_subnum = experiment_subnum
        self.experiment_episodes = 0
        self.instance_id = instance_id

        self.run_state = False

        self.performance = PerformanceSingle(parameters=self.p)

        if seed_offset is None:
            if self.p.experiment.generate_rand_seed_offset:
                self.p.experiment.seed_offset = int(time.time())
            elif self.p.experiment.seed_offset is None:
                self.p.experiment.seed_offset = 0
        else:
            self.p.experiment.seed_offset = seed_offset

        if self.p.experiment.type in [ExperimentType.EVAL_MULTI, ExperimentType.EVAL_SINGLE,
                                      ExperimentType.OPT_GRID_MULTI]:
            instance_offset = self.instance_id if self.instance_id is not None else 0
        else:
            instance_offset = 0
        np.random.seed(self.p.experiment.seed_offset + instance_offset)

        self.random_seed = self.p.experiment.seed_offset + instance_offset

    def load_params(self, experiment_type, experiment_id, experiment_num, instance_id, **kwargs):
        self.p_plot.load_default_params()
        if experiment_type == ExperimentType.OPT_GRID and instance_id > 0:
            self.p.load_experiment_params(experiment_type=ExperimentType.OPT_GRID, experiment_id=experiment_id,
                                          experiment_num=experiment_num, experiment_subnum=0,
                                          custom_params=kwargs)
        else:
            self.p.load_default_params(custom_params=kwargs)

            self.p.evaluate(parameters=self.p, recursive=True)

            if self.p.plasticity.tau_h is None:
                self.p.plasticity.tau_h = self.__compute_time_constant_dendritic_rate(dt_stm=self.p.encoding.dt_stm,
                                                                                      dt_seq=self.p.encoding.dt_seq,
                                                                                      target_firing_rate=self.p.plasticity.y
                                                                                      )

            # dynamically calculate new weights, scale by 1/1000 for "original" pynn-nest neurons
            if self.p.synapses.dyn_weight_calculation:
                self.p.synapses.w_ext_exc = psp_max_2_psc_max(self.p.synapses.j_ext_exc_psp,
                                                              self.p.neurons.excitatory.tau_m,
                                                              self.p.neurons.excitatory.tau_syn_ext,
                                                              self.p.neurons.excitatory.c_m) / 1000
                self.p.synapses.w_exc_inh = psp_max_2_psc_max(self.p.synapses.j_exc_inh_psp,
                                                              self.p.neurons.inhibitory.tau_m,
                                                              self.p.neurons.inhibitory.tau_syn_E,
                                                              self.p.neurons.inhibitory.c_m)
                self.p.synapses.w_inh_exc = abs(psp_max_2_psc_max(self.p.synapses.j_inh_exc_psp,
                                                                  self.p.neurons.excitatory.tau_m,
                                                                  self.p.neurons.excitatory.tau_syn_inh,
                                                                  self.p.neurons.excitatory.c_m)) / 1000

    def init_network(self):
        self.init_neurons()
        self.init_connections()
        self.init_external_input()

    def init_neurons(self):
        self.neurons_exc = self.init_all_neurons_exc()

        self.neurons_inh = self.init_neurons_inh()

        self.neurons_ext = Population(self.p.network.num_symbols, SpikeSourceArray())

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
        spike_times = [list() for _ in range(self.p.network.num_symbols)]
        spike_time = None

        sequence_offset = self.p.encoding.t_exc_start
        for _ in range(self.p.encoding.num_repetitions):
            for i_seq, sequence in enumerate(self.p.experiment.sequences):
                for i_element, element in enumerate(sequence):
                    spike_time = sequence_offset + i_element * self.p.encoding.dt_stm
                    spike_times[SYMBOLS[element]].append(spike_time)
                sequence_offset = spike_time + self.p.encoding.dt_seq

        self.last_ext_spike_time = spike_time

        log.debug(f'Spike times:')
        for i_letter, letter_spikes in enumerate(spike_times):
            log.debug(f'{list(SYMBOLS.keys())[i_letter]}: {spike_times[i_letter]}')

        self.neurons_ext.set(spike_times=spike_times)

        self.spike_times_ext = spike_times

        if init_performance:
            log.info(f'Initialized external input for sequence(s) {self.p.experiment.sequences}')
            # Initialize performance containers
            self.performance.init_data()

    def init_connections(self, exc_to_exc=None, exc_to_inh=None):
        self.ext_to_exc = []
        for i in range(self.p.network.num_symbols):
            self.ext_to_exc.append(Projection(
                PopulationView(self.neurons_ext, [i]),
                self.get_neurons(NeuronType.Soma, symbol_id=i),
                AllToAllConnector(),
                synapse_type=StaticSynapse(weight=self.p.synapses.w_ext_exc, delay=self.p.synapses.delay_ext_exc),
                receptor_type=self.p.synapses.receptor_ext_exc))

        self.exc_to_exc = []
        num_connections = int(self.p.network.num_neurons * self.p.synapses.p_exc_exc)
        i_w = 0
        for i in range(self.p.network.num_symbols):
            for j in range(self.p.network.num_symbols):
                if i == j:
                    i_w += 1
                    continue
                weight = self.p.synapses.w_exc_exc if exc_to_exc is None else exc_to_exc[i_w]
                seed = j + i * self.p.network.num_symbols + self.p.experiment.seed_offset
                if self.instance_id is not None:
                    seed += self.instance_id * self.p.network.num_symbols ** 2
                self.exc_to_exc.append(Projection(
                    self.get_neurons(NeuronType.Soma, symbol_id=i),
                    self.get_neurons(NeuronType.Dendrite, symbol_id=j),
                    FixedNumberPreConnector(num_connections, rng=NumpyRNG(seed=j + i * self.p.network.num_symbols)),
                    synapse_type=StaticSynapse(weight=weight, delay=self.p.synapses.delay_exc_exc),
                    receptor_type=self.p.synapses.receptor_exc_exc,
                    label=f"exc-exc_{id_to_symbol(i)}>{id_to_symbol(j)}"))
                i_w += 1

        self.exc_to_inh = []
        for i in range(self.p.network.num_symbols):
            weight = self.p.synapses.w_exc_inh if exc_to_inh is None else exc_to_inh[i]
            self.exc_to_inh.append(Projection(
                self.get_neurons(NeuronType.Soma, symbol_id=i),
                PopulationView(self.neurons_inh, [i]),
                AllToAllConnector(),
                synapse_type=StaticSynapse(weight=weight, delay=self.p.synapses.delay_exc_inh),
                receptor_type=self.p.synapses.receptor_exc_inh))

        self.inh_to_exc = []
        for i in range(self.p.network.num_symbols):
            self.inh_to_exc.append(Projection(
                PopulationView(self.neurons_inh, [i]),
                self.get_neurons(NeuronType.Soma, symbol_id=i),
                AllToAllConnector(),
                synapse_type=StaticSynapse(weight=self.p.synapses.w_inh_exc, delay=self.p.synapses.delay_inh_exc),
                receptor_type=self.p.synapses.receptor_inh_exc))

    def init_prerun(self):
        pass

    def init_backend(self, **kwargs):
        pass

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

        t_exc = (((len(self.p.experiment.sequences[0]) - 1) * dt_stm + dt_seq + calibration)
                 * len(self.p.experiment.sequences))

        log.debug("\nDuration of a sequence set %d ms" % t_exc)

        return target_firing_rate * t_exc

    @abstractmethod
    def reset(self, store_to_cache=False):
        pass

    def run_sim(self, runtime):
        pynn.run(runtime)
        self.run_state = True

    @abstractmethod
    def get_neurons(self, neuron_type, symbol_id=None):
        pass

    @abstractmethod
    def get_neuron_data(self, neuron_type, neurons=None, value_type="spikes", symbol_id=None, neuron_id=None,
                        runtime=None, dtype=None):
        pass

    def plot_events(self, neuron_types="all", symbols="all", size=None, x_lim_lower=None, x_lim_upper=None, seq_start=0,
                    seq_end=None, fig_title="", file_path=None, run_id=None, show_grid=False, separate_seqs=False):
        if size is None:
            size = (12, 10)

        if type(neuron_types) is str and neuron_types == "all":
            neuron_types = [NeuronType.Dendrite, NeuronType.Soma, NeuronType.Inhibitory]
        elif type(neuron_types) is list:
            pass
        else:
            return

        if run_id is None:
            max_time = self.p.experiment.runtime
        else:
            single_run_length = self.calc_runtime_single() - self.p.encoding.t_exc_start
            x_lim_lower = run_id * single_run_length
            x_lim_upper = (run_id + 1) * single_run_length - self.p.encoding.dt_seq * 0.9
            max_time = x_lim_upper

        if x_lim_lower is None:
            x_lim_lower = 0.
        if x_lim_upper is None:
            x_lim_upper = max_time

        if type(symbols) is str and symbols == "all":
            symbols = range(self.p.network.num_symbols)
        elif type(symbols) is list:
            pass

        n_cols = 1 if not separate_seqs else len(self.p.experiment.sequences)
        if len(symbols) == 1:
            fig, axs = plt.subplots(nrows=1, ncols=n_cols, figsize=size)
        else:
            fig, axs = plt.subplots(self.p.network.num_symbols, n_cols, sharex="col", sharey="row", figsize=size)

        if seq_end is None:
            seq_end = seq_start + self.p.experiment.runtime

        ax = None

        for i_seq in range(n_cols):
            if n_cols > 1:
                x_lim_upper = x_lim_lower + (len(self.p.experiment.sequences[0]) - 0.5) * self.p.encoding.dt_stm + self.p.encoding.t_exc_start
            for i_symbol in symbols:

                if len(symbols) == 1:
                    ax = axs[i_seq]
                elif n_cols == 1:
                    ax = axs[i_symbol]
                else:
                    ax = axs[i_symbol, i_seq]

                for neurons_i in neuron_types:
                    # Retrieve and plot spikes from selected neurons
                    spikes = deepcopy(self.neuron_events[neurons_i][i_symbol])
                    if neurons_i == NeuronType.Inhibitory:
                        spikes.append([])
                    else:
                        spikes.insert(0, [])
                    if neurons_i == NeuronType.Dendrite:
                        spikes_post = deepcopy(self.neuron_events[NeuronType.Soma][i_symbol])
                        plot_dendritic_events(ax, spikes[1:], spikes_post,
                                              tau_dap=self.p.neurons.dendrite.tau_dAP*self.p.encoding.t_scaling_factor,
                                              color=f"C{neurons_i.ID}", label=neurons_i.NAME.capitalize(),
                                              seq_start=seq_start, seq_end=seq_end, epoch_end=x_lim_upper)
                    else:
                        line_widths = 1.5
                        line_lengths = 1

                        ax.eventplot(spikes, linewidths=line_widths, linelengths=line_lengths,
                                     label=neurons_i.NAME.capitalize(), color=f"C{neurons_i.ID}")

                # plot external spikes as reference lines
                for spike_time_ext_sym_i in self.spike_times_ext[i_symbol]:
                    ax.plot([spike_time_ext_sym_i, spike_time_ext_sym_i], [0.6, self.p.network.num_neurons+0.4], c="grey",
                            label="External")

                # Configure the plot layout
                ax.set_xlim(x_lim_lower, x_lim_upper)
                ax.set_ylim(-1, self.p.network.num_neurons + 1)

                if i_seq < 1:
                    ax.yaxis.set_ticks(range(self.p.network.num_neurons + 2))
                    ax.set_ylabel(id_to_symbol(i_symbol), weight='bold',
                                  fontsize=self.p_plot.events.fontsize.subplot_labels)

                    # Generate y-tick-labels based on number of neurons per symbol
                    y_tick_labels = ['Inh', '', '0'] + ['' for _ in range(self.p.network.num_neurons - 2)] + [
                        str(self.p.network.num_neurons - 1)]
                    ax.set_yticklabels(y_tick_labels, rotation=45, fontsize=self.p_plot.events.fontsize.tick_labels)

                if show_grid:
                    ax.grid(True, which='both', axis='both')

                if (x_lim_upper - x_lim_lower) / self.p.encoding.dt_stm > 200:
                    log.info("Minor ticks not set because the number of ticks would be too high.")
                elif (x_lim_upper - x_lim_lower) / self.p.encoding.dt_stm < 15:
                    ax.xaxis.set_ticks(np.arange(x_lim_lower, x_lim_upper, self.p.encoding.dt_stm / 2))

            ax.tick_params(axis='x', labelsize=self.p_plot.events.fontsize.tick_labels)

            x_lim_lower += (len(self.p.experiment.sequences[0]) - 1) * self.p.encoding.dt_stm + self.p.encoding.dt_seq



        if n_cols > 1:
            plt.subplots_adjust(wspace=self.p_plot.events.padding.w_space)

        fig.text(0.5, 0.01, "Time [ms]", ha="center", fontsize=self.p_plot.events.fontsize.axis_labels)

        # Create custom legend for all plots
        custom_lines = [Line2D([0], [0], color=f"C{n.ID}", label=n.NAME.capitalize(), lw=3) for n in neuron_types]
        custom_lines.append(Line2D([0], [0], color=f"grey", label="External", lw=3))

        plt.figlegend(handles=custom_lines, loc=(0.715, 0.904), ncol=2, labelspacing=0.2,
                      fontsize=self.p_plot.events.fontsize.legend, fancybox=True,
                      borderaxespad=4)

        fig.text(0.05, 0.5, "Symbol & Neuron ID", va="center", rotation="vertical",
                 fontsize=self.p_plot.events.fontsize.axis_labels)

        fig.suptitle(fig_title, x=0.39, y=0.95, fontsize=self.p_plot.events.fontsize.title)
        fig.show()

        if file_path is not None:
            plt.savefig(f"{file_path}.pdf")

            pickle.dump(fig, open(f'{file_path}.fig.pickle',
                                  'wb'))  # This is for Python 3 - py2 may need `file` instead of `open`

    def plot_v_exc(self, alphabet_range, neuron_range='all', size=None, neuron_type=NeuronType.Soma, runtime=None,
                   show_legend=False, file_path=None):
        if size is None:
            size = (12, 10)

        if type(neuron_range) is str and neuron_range == 'all':
            neuron_range = range(self.p.network.num_neurons)
        elif type(neuron_range) is list or type(neuron_range) is range:
            pass
        else:
            return

        if type(runtime) is str:
            if str(runtime).lower() == 'max':
                runtime = self.last_ext_spike_time + (self.p.encoding.dt_seq - self.p.encoding.t_exc_start)
        elif type(runtime) is float or type(runtime) is int:
            pass
        else:
            runtime = self.p.experiment.runtime

        spike_times = [[]]
        header_spikes = list()

        fig, ax = plt.subplots(figsize=size)

        for alphabet_id in alphabet_range:
            # retrieve and save spike times
            spikes = self.neuron_events[neuron_type][alphabet_id]
            for neuron_id in neuron_range:
                # add spikes to list for printing
                spike_times[0].append(np.array(spikes[neuron_id]).round(5).tolist())
                header_spikes.append(f"{id_to_symbol(alphabet_id)}[{neuron_id}]")

                # retrieve voltage data
                data_v = self.get_neuron_data(neuron_type, value_type=RecTypes.V, symbol_id=alphabet_id,
                                              neuron_id=neuron_id, runtime=runtime)

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

    def plot_performance(self, statistic=StatisticalMetrics.MEAN, sequences="statistic", plot_dd=False):
        self.performance.plot(self.p_plot, statistic=statistic, sequences=sequences, plot_dd=plot_dd)

    def getsize(self):
        """sum size of object & members."""
        if isinstance(self, BLACKLIST):
            raise TypeError('getsize() does not take argument of type: ' + str(type(self)))
        seen_ids = set()
        size = 0
        objects = [self]
        while objects:
            need_referents = []
            for obj in objects:
                if not isinstance(obj, BLACKLIST) and id(obj) not in seen_ids:
                    seen_ids.add(id(obj))
                    size += sys.getsizeof(obj)
                    need_referents.append(obj)
            objects = get_referents(*need_referents)
        return size

    def calc_runtime_single(self):
        runtime_single = self.p.encoding.t_exc_start
        for seq in self.p.experiment.sequences:
            for element in seq[:-1]:
                runtime_single += self.p.encoding.dt_stm
            runtime_single += self.p.encoding.dt_seq
        return runtime_single

    def calc_num_correlated_events(self, neuron_type_a, neuron_type_b, symbol_id, t_max, t_min=0):
        """
        Calculate number of correlated events within neurons of symbol_id

        :param neuron_type_a:
        :type neuron_type_a:
        :param neuron_type_b:
        :type neuron_type_b:
        :param symbol_id:
        :type symbol_id:
        :param t_max:
        :type t_max:
        :param t_min:
        :type t_min:
        :return:
        :rtype:
        """

        events_a = self.neuron_events[neuron_type_a][symbol_id]
        events_b = self.neuron_events[neuron_type_b][symbol_id]

        num_events = np.zeros(self.p.network.num_neurons)
        for i_neuron, events_a_i in enumerate(events_a):
            events_b_i = events_b[i_neuron]
            for event_a_i in events_a_i:
                if event_a_i > self.p.plasticity.execution_start:
                    break
                diff_arr = events_b_i - event_a_i
                if len(np.where(np.logical_and(diff_arr > t_min, diff_arr < t_max))[0]) > 0:
                    num_events[i_neuron] += 1

        return num_events

    def __str__(self):
        return type(self).__name__


class SHTMTotal(SHTMBase, ABC):
    def __init__(self, experiment_type=ExperimentType.EVAL_SINGLE, experiment_id=None, experiment_num=None,
                 experiment_subnum=None, plasticity_cls=None, instance_id=None, seed_offset=None, p=None, **kwargs):
        super().__init__(experiment_type=experiment_type, experiment_id=experiment_id, experiment_num=experiment_num,
                         experiment_subnum=experiment_subnum, instance_id=instance_id, seed_offset=seed_offset, p=p,
                         **kwargs)

        self.con_plastic = None
        self.trace_dendrites = self.trace_dendrites = np.zeros(shape=(self.p.network.num_symbols,
                                                                      self.p.network.num_neurons))

        if plasticity_cls is None:
            self.plasticity_cls = Plasticity
        else:
            self.plasticity_cls = plasticity_cls

        if self.p.experiment.log_permanence is None or not self.p.experiment.log_permanence:
            self.log_permanence = list()
            self.p.experiment.log_permanence = False
        else:
            self.log_permanence = range(self.p.network.num_symbols ** 2 - self.p.network.num_symbols)

        if self.p.experiment.log_weights is None or not self.p.experiment.log_weights:
            self.log_weights = list()
            self.p.experiment.log_weights = None
        else:
            self.log_weights = range(self.p.network.num_symbols ** 2 - self.p.network.num_symbols)

    def init_connections(self, exc_to_exc=None, exc_to_inh=None, debug=False):
        super().init_connections()

        self.con_plastic = list()

        for i_plastic in range(len(self.exc_to_exc)):
            # Retrieve id (letter) of post synaptic neuron population
            symbol_post = self.exc_to_exc[i_plastic].label.split('_')[1].split('>')[1]
            # Create population view of all post synaptic somas
            post_somas = PopulationView(self.get_neurons(NeuronType.Soma, symbol_id=SYMBOLS[symbol_post]),
                                        list(range(self.p.network.num_neurons)))
            if self.p.synapses.dyn_inh_weights:
                proj_post_soma_inh = self.exc_to_inh[SYMBOLS[symbol_post]]
            else:
                proj_post_soma_inh = None

            self.con_plastic.append(self.plasticity_cls(self.exc_to_exc[i_plastic], post_somas=post_somas, shtm=self,
                                                        proj_post_soma_inh=proj_post_soma_inh, index=i_plastic,
                                                        debug=debug, **self.p.plasticity.dict()))

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

        for i_plot, i_con in enumerate(plot_con_ids):
            permanences = np.array(self.con_plastic[i_con].permanences)

            ind = np.where(np.sum(permanences == 0, axis=0) == 0)[0]
            permanences = permanences[:, ind].tolist()

            permanences_plot = list()
            for i_perm in range(len(permanences)):
                if not np.equal(permanences[i_perm], 0).all():
                    permanences_plot.append(permanences[i_perm])

            # Plot all previous permanences as a line over time
            axs[i_plot].plot(range(len(permanences_plot)), permanences_plot)

            if '>' in self.con_plastic[i_con].projection.label:
                y_label = self.con_plastic[i_con].projection.label.split('_')[1]
            else:
                i_con_total = i_con + int(i_con > 3) + int(i_con > 7) + 1
                y_label = (f"{id_to_symbol(int(i_con_total / self.p.network.num_symbols))}>"
                           f"{id_to_symbol(int(i_con_total % self.p.network.num_symbols))}")
            axs[i_plot].set_ylabel(y_label, weight='bold')
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

    def _retrieve_neuron_data(self):
        self.neuron_events = dict()

        for neuron_type in NeuronType.get_all_types():
            self.neuron_events[neuron_type] = list()
            for i_symbol in range(self.p.network.num_symbols):
                events = self.get_neuron_data(neuron_type, value_type=RecTypes.SPIKES, symbol_id=i_symbol,
                                              dtype=list)
                self.neuron_events[neuron_type].append(events)

    def get_spike_times(self, runtime, dt):
        log.detail("Calculating spike times")

        times = np.linspace(0., runtime, int(runtime / dt))

        spike_times_dendrite = np.zeros((self.p.network.num_symbols, self.p.network.num_neurons, len(times)),
                                        dtype=np.int8)
        spike_times_soma = np.zeros((self.p.network.num_symbols, self.p.network.num_neurons, len(times)), dtype=np.int8)

        for i_symbol in range(self.p.network.num_symbols):
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
        for i_symbol in range(self.p.network.num_symbols):
            for i_neuron in range(self.p.network.num_neurons):
                events = np.array(self.neuron_events[NeuronType.Dendrite][i_symbol][i_neuron])
                self.trace_dendrites[i_symbol, i_neuron] = calculate_trace(self.trace_dendrites[i_symbol, i_neuron],
                                                                           0, self.p.experiment.runtime,
                                                                           events, self.p.plasticity.tau_h)

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

    def run(self, runtime=None, steps=None, plasticity_enabled=True, store_to_cache=False, dyn_exc_inh=False,
            run_type=RunType.SINGLE):
        if runtime is None:
            runtime = self.p.experiment.runtime
        if steps is None:
            steps = self.p.experiment.episodes
        self.p.experiment.episodes = 0

        if type(runtime) is str:
            if str(runtime).lower() == 'max':
                runtime = self.last_ext_spike_time + (self.p.encoding.dt_seq - self.p.encoding.t_exc_start)
        elif type(runtime) is float or type(runtime) is int:
            pass
        elif runtime is None:
            log.debug("No runtime specified. Setting runtime to last spike time + 2xdt_stm")
            runtime = self.last_ext_spike_time + (self.p.encoding.dt_seq - self.p.encoding.t_exc_start)
        else:
            log.error("Error! Wrong runtime")

        self.p.experiment.runtime = runtime

        for t in range(steps):
            log.info(f'Running emulation step {self.experiment_episodes + t + 1}/{self.experiment_episodes + steps}')

            # reset the simulator and the network state if not first run
            if self.run_state:
                self.reset(store_to_cache)

            # set start time to 0.0 because
            # - nest is reset and always starts with 0.0
            # - bss2 resets the time itself after each run to 0.0
            sim_start_time = 0.0
            log.detail(f"Current time: {sim_start_time}")

            self.run_sim(runtime)

            self._retrieve_neuron_data()

            if self.p.performance.compute_performance:
                self.performance.compute(neuron_events=self.neuron_events, method=self.p.performance.method)

            if plasticity_enabled:
                if run_type == RunType.MULTI:
                    log.warn(
                        f"Multi-core version of plasticity calculation is currently not working. Please choose the "
                        f"single-core version. Not calculating plasticity.")
                    # self.__run_plasticity_parallel(runtime, sim_start_time, dyn_exc_inh=dyn_exc_inh)
                elif run_type == RunType.SINGLE:
                    self.__run_plasticity_singular(runtime, sim_start_time, dyn_exc_inh=dyn_exc_inh)

            if self.p.experiment.save_auto and self.p.experiment.save_auto_epoches > 0:
                if (t + 1) % self.p.experiment.save_auto_epoches == 0:
                    self.p.experiment.episodes = self.experiment_episodes + t + 1
                    self.save_full_state()

            if self.p.plasticity.learning_rate_decay is not None:
                self.p.plasticity.learning_factor *= self.p.plasticity.learning_rate_decay

        # print performance results
        self.print_performance_results()

        self.experiment_episodes += steps
        self.p.experiment.episodes = self.experiment_episodes

        if self.p.experiment.save_final or self.p.experiment.save_auto:
            self.save_full_state()

    def print_performance_results(self):
        performance_results = self.performance.get_performance_dict(final_result=True,
                                                                    running_avgs=self.p.performance.running_avgs)
        log.essens(f"Performance (0.5):  {performance_results['error_running-avg-0.5']}  |  "
                   f"Epochs:  {performance_results['num-epochs']}")


    def __run_plasticity_singular(self, runtime, sim_start_time, dyn_exc_inh=False):
        log.info("Starting plasticity calculations")

        active_synapse_post = np.zeros((self.p.network.num_symbols, self.p.network.num_neurons))

        # Calculate plasticity for each synapse
        for i_plasticity, plasticity in enumerate(self.con_plastic):
            plasticity(runtime, sim_start_time=sim_start_time)
            log.debug(f"Finished plasticity calculation {i_plasticity + 1}/{len(self.con_plastic)}")

            if dyn_exc_inh:
                w = self.exc_to_exc[i_plasticity].get("weight", format="array")
                letter_id = SYMBOLS[plasticity.get_post_symbol()]
                active_synapse_post[letter_id, :] = np.logical_or(active_synapse_post[letter_id, :],
                                                                  np.any(w > 0, axis=0))

        if dyn_exc_inh and self.p.synapses.w_exc_inh_dyn is not None:
            for i_inh in range(self.p.network.num_symbols):
                w = self.exc_to_inh.get("weight", format="array")
                w[active_synapse_post[i_inh, :]] = self.p.synapses.w_exc_inh_dyn

        self.__update_dendritic_trace()

    def __run_plasticity_parallel(self, runtime, sim_start_time, dyn_exc_inh=False):
        log.info("Starting plasticity calculations")

        active_synapse_post = np.zeros((self.p.network.num_symbols, self.p.network.num_neurons))

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

        if dyn_exc_inh and self.p.synapses.w_exc_inh_dyn is not None:
            for i_inh in range(self.p.network.num_symbols):
                w = self.exc_to_inh.get("weight", format="array")
                w[active_synapse_post[i_inh, :]] = self.p.synapses.w_exc_inh_dyn

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

    def save_config(self):
        folder_path = get_experiment_folder(self, self.p.experiment.type, self.p.experiment.id, self.experiment_num,
                                            experiment_subnum=self.experiment_subnum, instance_id=self.instance_id)
        file_path = join(folder_path, f"config_network.yaml")

        with open(file_path, 'w') as file:
            yaml.dump(self.p.dict(exclude_none=True), file)

    def save_performance_data(self):
        folder_path = get_experiment_folder(self, self.p.experiment.type, self.p.experiment.id, self.experiment_num,
                                            experiment_subnum=self.experiment_subnum, instance_id=self.instance_id)
        file_path = join(folder_path, "performance")

        np.savez(file_path, **self.performance.data)

    def save_network_data(self):
        # ToDo: Check if this works with bss2
        folder_path = get_experiment_folder(self, self.p.experiment.type, self.p.experiment.id, self.experiment_num,
                                            experiment_subnum=self.experiment_subnum, instance_id=self.instance_id)

        # Save weights
        file_path = join(folder_path, "weights")

        weights_dict = {var_name: getattr(self, var_name) for var_name in RuntimeConfig.saved_weights}
        for con_name, connections in weights_dict.items():
            weights_all = list()
            for connection in connections:
                weights_all.append(connection.get("weight", format="array"))
            weights_dict[con_name] = np.array(weights_all)

        np.savez(file_path, **weights_dict)

        # Save events
        file_path = join(folder_path, "events.pkl")
        with open(file_path, 'wb') as f:
            pickle.dump(self.neuron_events, f)

        # Save network variables
        file_path = join(folder_path, "network")

        network_dict = {var_name: getattr(self, var_name) for var_name in RuntimeConfig.saved_network_vars}

        np.savez(file_path, **network_dict)

        # Save plasticity parameters
        file_path = join(folder_path, "plasticity")

        plasticity_dict = {var_name: list() for var_name in RuntimeConfig.saved_plasticity_vars}
        for con_plastic in self.con_plastic:
            for var_name in plasticity_dict.keys():
                plasticity_dict[var_name].append(getattr(con_plastic, var_name))

        for var_name in plasticity_dict.keys():
            plasticity_dict[var_name] = np.array(plasticity_dict[var_name])

        np.savez(file_path, **plasticity_dict)

    def save_full_state(self, running_avg_perc=0.5, optimized_parameter_ranges=None, save_setup=False):
        log.debug("Saving full state of network and experiment.")

        if (self.p.experiment.type in
                [ExperimentType.EVAL_MULTI, ExperimentType.OPT_GRID, ExperimentType.OPT_GRID_MULTI]):
            if self.instance_id is not None and self.instance_id == 0 and save_setup:
                self.experiment_num = save_experimental_setup(net=self, experiment_num=self.experiment_num,
                                                              experiment_subnum=self.experiment_subnum,
                                                              instance_id=self.instance_id,
                                                              optimized_parameter_ranges=optimized_parameter_ranges)
            save_instance_setup(net=self.__str__(), parameters=self.p,
                                performance=self.performance.get_performance_dict(final_result=True,
                                                                                  running_avgs=self.p.performance.running_avgs,
                                                                                  decimals=3),
                                experiment_num=self.experiment_num, experiment_subnum=self.experiment_subnum,
                                instance_id=self.instance_id,
                                optimized_parameters=self.optimized_parameters)
        else:
            self.experiment_num = save_experimental_setup(net=self, experiment_num=self.experiment_num,
                                                          experiment_subnum=self.experiment_subnum,
                                                          instance_id=self.instance_id)

        self.save_config()
        self.save_performance_data()
        self.save_network_data()

    def load_network_data(self, experiment_type, experiment_num, experiment_subnum=None, instance_id=None):
        # ToDo: Check if this works with bss2
        folder_path = get_experiment_folder(self, experiment_type, self.p.experiment.id, experiment_num,
                                            experiment_subnum=experiment_subnum, instance_id=instance_id)

        # Load weights
        file_path = join(folder_path, "weights.npz")
        data_weights = np.load(file_path)

        # Load events
        file_path = join(folder_path, "events.pkl")
        with open(file_path, 'rb') as f:
            self.neuron_events = pickle.load(f)

        # Load network variables
        file_path = join(folder_path, "network.npz")
        data_network = np.load(file_path)
        for var_name, var_value in data_network.items():
            setattr(self, var_name, var_value)

        # Load plasticity parameters
        file_path = join(folder_path, "plasticity.npz")
        data_plasticity = np.load(file_path, allow_pickle=True)
        data_plasticity = dict(data_plasticity)
        for var_name in ["permanences", "weights"]:
            if var_name in data_plasticity.keys():
                data_plasticity[var_name] = data_plasticity[var_name].tolist()

        return data_weights, data_plasticity

    @staticmethod
    def load_full_state(network_type, experiment_id, experiment_num, experiment_type=ExperimentType.EVAL_SINGLE,
                        experiment_subnum=None, instance_id=None, debug=False):
        log.debug("Loading full state of network and experiment.")

        p = NetworkParameters(network_type=network_type)
        p.load_experiment_params(experiment_type=experiment_type, experiment_id=experiment_id,
                                 experiment_num=experiment_num, experiment_subnum=experiment_subnum,
                                 instance_id=instance_id)

        shtm = network_type(p=p)

        shtm.p_plot = PlottingParameters(network_type=network_type)
        shtm.p_plot.load_default_params()

        shtm.performance.load_data(shtm, experiment_type, experiment_id, experiment_num,
                                   experiment_subnum=experiment_subnum, instance_id=instance_id)
        data_weights, data_plasticity = shtm.load_network_data(experiment_type, experiment_num,
                                                               experiment_subnum=experiment_subnum,
                                                               instance_id=instance_id)

        shtm.init_neurons()
        shtm.init_connections(debug=debug)

        for i_con_plastic in range(len(shtm.con_plastic)):
            for var_name, var_value in data_plasticity.items():
                setattr(shtm.con_plastic[i_con_plastic], var_name, var_value[i_con_plastic])

        shtm.init_external_input()

        return shtm


class Plasticity(ABC):
    def __init__(self, projection: Projection, post_somas, shtm, index, proj_post_soma_inh=None, debug=False,
                 learning_factor=None, permanence_init_min=None, permanence_init_max=None, permanence_max=None,
                 permanence_threshold=None, w_mature=None, y=None, lambda_plus=None, weight_learning=None,
                 weight_learning_scale=None, lambda_minus=None, lambda_h=None, target_rate_h=None, tau_plus=None,
                 tau_h=None, delta_t_min=None, delta_t_max=None, dt=None, correlation_threshold=None,
                 homeostasis_depression_rate=None, **kwargs):
        # custom objects
        self.projection = projection
        self.proj_post_soma_inh = proj_post_soma_inh
        self.shtm: SHTMTotal = shtm
        self.post_somas = post_somas

        # editable/changing variables
        if permanence_init_min == permanence_init_max:
            self.permanence_min = np.ones(shape=(len(self.projection),), dtype=float) * permanence_init_min
        else:
            self.permanence_min = np.asarray(np.random.randint(permanence_init_min, permanence_init_max,
                                                               size=(len(self.projection),)), dtype=float)
        self.permanence = copy.copy(self.permanence_min)
        self.permanences = None
        self.weights = None
        self.weight_learning = None
        self.weight_learning_scale = None
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
        self.permanence_threshold = np.ones((len(self.projection))) * permanence_threshold
        self.correlation_threshold = correlation_threshold
        self.lambda_plus = lambda_plus * learning_factor
        self.lambda_minus = lambda_minus * learning_factor
        self.lambda_h = lambda_h * learning_factor
        self.homeostasis_depression_rate = homeostasis_depression_rate

        self.learning_rules = {"original": self.rule, "bss2": self.rule_bss2}

        self.symbol_id_pre = SYMBOLS[symbol_from_label(self.projection.label, ID_PRE)]
        self.symbol_id_post = SYMBOLS[symbol_from_label(self.projection.label, ID_POST)]

        self.connections = list()

        self.int_conversion = "int" in self.shtm.p.plasticity.type
        if self.int_conversion:
            self.scale_permanence(min_value=0, max_value=256)

    def scale_permanence(self, min_value, max_value):
        factor = max_value/self.permanence_max
        self.permanence_min = np.asarray(self.permanence_min * factor, dtype=int)
        self.permanence = copy.copy(self.permanence_min)

        self.permanence_max = int(max_value)
        self.permanence_threshold = np.ones((len(self.projection)), dtype=int) * int(max_value/2)

        # self.target_rate_h *= max_value
        # self.correlation_threshold *= max_value

    def rule(self, permanence, permanence_threshold, x, z, runtime, permanence_min,
             neuron_spikes_pre, neuron_spikes_post_soma, neuron_spikes_post_dendrite,
             delay, sim_start_time=0.0):
        last_spike_pre = 0

        neuron_spikes_pre = np.array(neuron_spikes_pre)
        neuron_spikes_post_soma = np.array(neuron_spikes_post_soma)
        neuron_spikes_post_dendrite = np.array(neuron_spikes_post_dendrite)

        permanence_before = permanence

        # log.debug(f"{self.id}  permanence before: {permanence}")

        # loop through pre-synaptic spikes
        for spike_pre in neuron_spikes_pre:
            # calculate temporary x/z value (pre-synaptic/post-dendritic decay)
            x = x * np.exp(-(spike_pre - last_spike_pre) / self.tau_plus) + 1.0

            # loop through post-synaptic spikes between pre-synaptic spikes
            for spike_post in neuron_spikes_post_soma:
                spike_dt = (spike_post + delay) - spike_pre

                # log.debug(f"{self.id}  spikes: {spike_pre}, {spike_post}, {spike_dt}")

                # check if spike-dif is in boundaries
                if self.delta_t_min < spike_dt < self.delta_t_max:
                    # calculate temporary x value (pre synaptic decay)
                    x_tmp = x * np.exp(-spike_dt / self.tau_plus)
                    z_tmp = calculate_trace(z, sim_start_time, spike_post, neuron_spikes_post_dendrite,
                                            self.tau_h)

                    # hebbian learning
                    permanence = self.__facilitate(permanence, x_tmp)
                    d_facilitate = permanence - permanence_before
                    # log.debug(f"{self.id}  d_permanence facilitate: {d_facilitate}")
                    permanence_before = permanence

                    permanence = self.__homeostasis_control(permanence, z_tmp, permanence_min)
                    # if permanence - permanence_before < 0:
                    #     log.debug(f"{self.id}  d_permanence homeostasis: {permanence - permanence_before}")
                    # if self.debug and permanence - permanence_before < 0:
                        # log.info(f"{self.id}  spikes: {spike_pre}, {spike_post}, {spike_dt}")
                        # log.info(f"{self.id}  d_permanence facilitate: {d_facilitate}")
                        # log.info(f"{self.id}  d_permanence homeostasis: {permanence - permanence_before}")
                    permanence_before = permanence

            permanence = self.__depress(permanence, permanence_min)
            last_spike_pre = spike_pre
            # log.debug(f"{self.id}  permanence depression: {permanence - permanence_before}")

        # log.debug(f"{self.id}  permanence after: {permanence}")

        # update x (kplus) and z
        x = x * np.exp(-(runtime - last_spike_pre) / self.tau_plus)

        if permanence >= permanence_threshold:
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

    def __to_int(self, value, precision=256, amplitude=1.5):
        return int(value * precision)

    def rule_bss2(self, permanence, permanence_threshold, x, z, runtime, permanence_min,
                  neuron_spikes_pre, neuron_spikes_post_soma, neuron_spikes_post_dendrite,
                  delay, sim_start_time=0.0):
        neuron_spikes_pre = np.array(neuron_spikes_pre)
        neuron_spikes_post_soma = np.array(neuron_spikes_post_soma)
        neuron_spikes_post_dendrite = np.array(neuron_spikes_post_dendrite)

        permanence_before = permanence

        # log.debug(f"{self.id}  permanence before: {permanence}")

        x = 0
        z_tmp = 0

        # Calculate accumulated x
        spike_pairs_soma_soma = 0
        for spike_pre in neuron_spikes_pre:
            for spike_post in neuron_spikes_post_soma:
                spike_dt = spike_post - spike_pre

                # ToDo: Update rule based on actual trace calculation from BSS-2
                if spike_dt >= 0:
                    spike_pairs_soma_soma += 1
                    # log.debug(f"{self.id}  spikes (ss): {spike_pre}, {spike_post}, {spike_dt}")
                    # if self.int_conversion:
                    #     x += self.__to_int(np.exp(-spike_dt / self.tau_plus))
                    # else:
                    x += np.exp(-spike_dt / self.tau_plus)


        # Calculate accumulated z
        spike_pairs_dend_soma = 0
        for spike_post_dendrite in neuron_spikes_post_dendrite:
            for spike_post in neuron_spikes_post_soma:
                spike_dt = spike_post - spike_post_dendrite

                # ToDo: Update rule based on actual trace calculation from BSS-2
                if spike_dt >= 0:
                    spike_pairs_dend_soma += 1
                    # log.debug(f"{self.id}  spikes (ds): {spike_post_dendrite}, {spike_post}, {spike_dt}")
                    # if self.int_conversion:
                    #     z_tmp += self.__to_int(np.exp(-spike_dt / self.tau_plus))
                    # else:
                    z_tmp += np.exp(-spike_dt / self.tau_plus)

        # if z_tmp > 0:
        #     log.debug(f"{self.id}  x: {x},  z: {z_tmp}")

        # Calculation of z based on x
        # z = np.exp(-(-self.tau_plus*z_mean)/self.tau_h) * spike_pairs_dend_soma
        # Calculation of z using only number of pre-post spike pairs
        # z = spike_pairs_dend_soma
        # Calculation of z according to current BSS-2 implementation
        z = z_tmp

        # trace_threshold = np.exp(-self.delta_t_max / self.tau_plus)

        # log.debug(f"{self.id} permanence_threshold: {trace_threshold}")
        # log.debug(f"{self.id} x: {x},   x_mean: {x_mean}")
        # log.debug(f"{self.id} z: {z},   z_mean: {z_mean}")

        # hebbian learning
        # Only run facilitate/homeostasis if a spike pair exists with a delta within boundaries,
        # i.e. x or z > 0
        if x > self.correlation_threshold:
            permanence = self.__facilitate_bss2(permanence, x)
        log.debug(f"{self.id}  d_permanence facilitate: {permanence - permanence_before}")
        permanence_before = permanence

        # if self.int_conversion:
        #     permanence = int(np.round(permanence))

        if x > self.correlation_threshold:
            permanence = self.__homeostasis_control_bss2(permanence, z, permanence_min)
        # if permanence - permanence_before < 0:
        # if z > 0:
        log.debug(f"{self.id}  d_permanence homeostasis: {permanence - permanence_before}")
        permanence_before = permanence

        # if self.int_conversion:
        #     permanence = int(np.round(permanence))

        permanence = self.__depress_bss2(permanence, permanence_min, num_spikes=len(neuron_spikes_pre))
        log.debug(f"{self.id}  permanence depression: {permanence - permanence_before}")

        if self.int_conversion:
            permanence = int(np.round(permanence))

        # log.debug(f"{self.id}  permanence after: {permanence}")

        return permanence, x, permanence >= permanence_threshold

    def __facilitate_bss2(self, permanence, x):
        d_permanence = self.lambda_plus * x
        # if not self.int_conversion:
        d_permanence *= self.permanence_max
        permanence += d_permanence
        return min(permanence, self.permanence_max)

    def __homeostasis_control_bss2(self, permanence, z, permanence_min):
        z_prime = self.target_rate_h - z
        # if self.int_conversion:
        #     z_prime /= 256
        if z_prime < 0:
            z_prime = -np.exp(-z_prime/self.homeostasis_depression_rate)

        permanence = permanence + self.lambda_h * z_prime * self.permanence_max
        return max(min(permanence, self.permanence_max), permanence_min)

    def __depress_bss2(self, permanence, permanence_min, num_spikes):
        permanence = permanence - self.lambda_minus * self.permanence_max * num_spikes
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

    def init_connections(self):
        for c, connection in enumerate(self.get_connections()):
            i = self.get_connection_id_post(connection)
            j = self.get_connection_id_pre(connection)
            self.connections.append([c, j, i])

    def __call__(self, runtime: float, sim_start_time=0.0, q_plasticity=None):
        if self.connections is None or len(self.connections) <= 0:
            self.init_connections()

        spikes_pre = self.shtm.neuron_events[NeuronType.Soma][self.symbol_id_pre]
        spikes_post_dendrite = self.shtm.neuron_events[NeuronType.Dendrite][self.symbol_id_post]
        spikes_post_soma = self.shtm.neuron_events[NeuronType.Soma][self.symbol_id_post]

        weight = self.projection.get("weight", format="array")
        weight_before = np.copy(weight)

        for c, j, i in self.connections:
            neuron_spikes_pre = spikes_pre[j]
            neuron_spikes_post_dendrite = spikes_post_dendrite[i]
            neuron_spikes_post_soma = spikes_post_soma[i]
            z = self.shtm.trace_dendrites[self.symbol_id_post, i]

            # if self.debug:
            #     log.debug(f"Permanence calculation for connection {c} [{i}, {j}]")
            #     log.debug(f"Spikes pre [soma]: {neuron_spikes_pre}")
            #     log.debug(f"Spikes post [dend]: {neuron_spikes_post_dendrite}")
            #     log.debug(f"Spikes post [soma]: {neuron_spikes_post_soma}")

            permanence, x, mature = (self.learning_rules[self.shtm.p.plasticity.type.split("_")[0]]
                                     (permanence=self.permanence[c],
                                      permanence_threshold=self.permanence_threshold[c],
                                      runtime=runtime, x=self.x[j], z=z,
                                      permanence_min=self.permanence_min[c],
                                      neuron_spikes_pre=neuron_spikes_pre,
                                      neuron_spikes_post_soma=neuron_spikes_post_soma,
                                      neuron_spikes_post_dendrite=neuron_spikes_post_dendrite,
                                      delay=self.shtm.p.synapses.delay_exc_exc,
                                      sim_start_time=sim_start_time))

            self.permanence[c] = permanence
            self.x[j] = x

            if mature:
                weight_offset = (
                                            permanence - self.permanence_threshold) * self.weight_learning_scale if self.weight_learning else 0
                weight[j, i] = self.w_mature + weight_offset
                if self.proj_post_soma_inh is not None:
                    weight_inh = self.proj_post_soma_inh.get("weight", format="array")
                    weight_inh[i, :] = 250
                    # log.debug(f"+ | W_inh[{i}] = {weight_inh.flatten()}")
                    self.proj_post_soma_inh.set(weight=weight_inh)
            else:
                weight[j, i] = 0
                if self.proj_post_soma_inh is not None:
                    weight_inh = self.proj_post_soma_inh.get("weight", format="array")
                    weight_inh_old = np.copy(weight_inh)
                    weight_inh[i, :] = 0
                    # if np.sum(weight_inh_old.flatten() - weight_inh.flatten()) == 0:
                    #     log.debug(f"- | W_inh[{i}] = {weight_inh.flatten()}")
                    self.proj_post_soma_inh.set(weight=weight_inh)

        weight_diff = weight-weight_before
        if np.logical_and(weight_diff != 0, ~np.isnan(weight_diff)).any():
            self.projection.set(weight=weight)

        if self.permanences is not None:
            self.permanences.append(np.copy(np.round(self.permanence, 6)))
        if self.weights is not None:
            self.weights.append(
                np.copy(np.round(self.projection.get("weight", format="array").flatten(), 6)))

        log.debug(f'Finished execution of plasticity for {self.id}')

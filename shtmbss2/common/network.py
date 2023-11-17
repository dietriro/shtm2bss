import numpy as np
import copy
import pickle

from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
from pyNN.random import NumpyRNG
from tabulate import tabulate
from abc import ABC, abstractmethod

from shtmbss2.common.config import *
from shtmbss2.core.logging import log
from shtmbss2.core.parameters import Parameters
from shtmbss2.common.plot import plot_dendritic_events

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


def symbol_from_label(label, endpoint):
    return label.split('_')[1].split('>')[endpoint]


class NeuronType:
    class Dendrite:
        ID = 0
        NAME = "dendrite"

    class Soma:
        ID = 1
        NAME = "soma"

    class Inhibitory:
        ID = 2
        NAME = "inhibitory"


class RecTypes:
    SPIKES = "spikes"
    V = "v"


class SHTMBase(ABC):
    ALPHABET = {'A': 0,
                'B': 1,
                'C': 2,
                'D': 3,
                'E': 4,
                'F': 5,
                'G': 6,
                'H': 7}

    def __init__(self, **kwargs):
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

    def load_params(self, **kwargs):
        self.p = Parameters(network_type=self, custom_params=kwargs)

    def init_network(self):
        self.init_neurons()
        self.init_connections()
        self.init_external_input()
        self.init_rec_exc()

    def init_neurons(self):
        self.neurons_exc = self.init_all_neurons_exc()

        self.neurons_inh = self.init_neurons_inh()

        self.neurons_ext =  Population(self.p.Network.num_symbols, SpikeSourceArray())

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

    def init_external_input(self, init_recorder=False):
        spike_times = [list() for i in range(self.p.Network.num_symbols)]
        spike_time = None

        sequence_offset = 0
        for i_rep in range(self.p.Experiment.seq_repetitions):
            for i_seq, sequence in enumerate(self.p.Experiment.sequences):
                for i_element, element in enumerate(sequence):
                    spike_time = sequence_offset + i_element * self.p.Encoding.dt_stm + self.p.Encoding.dt_stm
                    spike_times[self.ALPHABET[element]].append(spike_time)
                sequence_offset = spike_time + self.p.Encoding.dt_seq

        self.last_ext_spike_time = spike_time

        log.info(f'Initialized external input for sequence(s) {self.p.Experiment.sequences}')
        log.debug(f'Spike times:')
        for i_letter, letter_spikes in enumerate(spike_times):
            log.debug(f'{list(self.ALPHABET.keys())[i_letter]}: {spike_times[i_letter]}')

        self.neurons_ext.set(spike_times=spike_times)

    def init_connections(self):
        self.ext_to_exc = []
        for i in range(self.p.Network.num_symbols):
            self.ext_to_exc.append(Projection(
                PopulationView(self.neurons_ext, [i]),
                self.get_neurons(NeuronType.Soma, symbol_id=i),
                AllToAllConnector(),
                synapse_type=StaticSynapse(weight=self.p.Synapses.w_ext_exc),
                receptor_type=self.p.Synapses.receptor_ext_exc))

        self.exc_to_exc = []
        num_connections = int(self.p.Network.num_neurons * self.p.Synapses.p_exc_exc)
        for i in range(self.p.Network.num_symbols):
            for j in range(self.p.Network.num_symbols):
                if i == j:
                    continue
                self.exc_to_exc.append(Projection(
                    self.get_neurons(NeuronType.Soma, symbol_id=i),
                    self.get_neurons(NeuronType.Dendrite, symbol_id=j),
                    FixedNumberPreConnector(num_connections, rng=NumpyRNG(seed=j + i * self.p.Network.num_symbols)),
                    synapse_type=StaticSynapse(weight=self.p.Synapses.w_exc_exc),
                    receptor_type=self.p.Synapses.receptor_exc_exc,
                    label=f"exc-exc_{self.id_to_letter(i)}>{self.id_to_letter(j)}"))

        self.exc_to_inh = []
        for i in range(self.p.Network.num_symbols):
            self.exc_to_inh.append(Projection(
                self.get_neurons(NeuronType.Soma, symbol_id=i),
                PopulationView(self.neurons_inh, [i]),
                AllToAllConnector(),
                synapse_type=StaticSynapse(weight=self.p.Synapses.w_exc_inh),
                receptor_type=self.p.Synapses.receptor_exc_inh))

        self.inh_to_exc = []
        for i in range(self.p.Network.num_symbols):
            self.inh_to_exc.append(Projection(
                PopulationView(self.neurons_inh, [i]),
                self.get_neurons(NeuronType.Soma, symbol_id=i),
                AllToAllConnector(),
                synapse_type=StaticSynapse(weight=self.p.Synapses.w_inh_exc),
                receptor_type=self.p.Synapses.receptor_inh_exc))

    def reset(self):
        pass

    @abstractmethod
    def get_neurons(self, neuron_type, symbol_id=None):
        pass

    @abstractmethod
    def get_neuron_data(self, neuron_type, neurons=None, value_type="spikes", symbol_id=None, neuron_id=None,
                        runtime=None):
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
                # neurons = PopulationView(self.neurons_inh, [i_symbol]) if neurons_i == NeuronType.Inhibitory else self.neurons_exc[i_symbol]
                # Retrieve and plot spikes from selected neurons
                spikes = [s.base for s in self.get_neuron_data(neuron_type=neurons_i,
                                                               # neurons=neurons,
                                                               symbol_id=i_symbol,
                                                               value_type=RecTypes.SPIKES)]
                if neurons_i == NeuronType.Inhibitory:
                    spikes.append([])
                else:
                    spikes.insert(0, [])
                if neurons_i == NeuronType.Dendrite:
                    spikes_post = [s.base for s in self.get_neuron_data(neuron_type=NeuronType.Soma,
                                                                        # neurons=neurons,
                                                                        symbol_id=i_symbol,
                                                                        value_type=RecTypes.SPIKES)]
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
            ax.set_ylabel(self.id_to_letter(i_symbol), weight='bold', fontsize=20)
            # ax.grid(True, which='both', axis='both')

            # Generate y-tick-labels based on number of neurons per symbol
            y_tick_labels = ['Inh', '', '0'] + ['' for k in range(self.p.Network.num_neurons - 2)] + [
                str(self.p.Network.num_neurons - 1)]
            ax.set_yticklabels(y_tick_labels, rotation=45, fontsize=18)

        # Create custom legend for all plots
        custom_lines = [Line2D([0], [0], color=f"C{n.ID}", label=n.NAME.capitalize(), lw=3) for n in neuron_types]

        ax.set_xlabel("Time [ms]", fontsize=26, labelpad=14)
        ax.xaxis.set_ticks(np.arange(x_lim_lower, x_lim_upper, self.p.Encoding.dt_stm/2))
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
            spikes = self.get_neuron_data(neuron_type, value_type=RecTypes.SPIKES, symbol_id=alphabet_id)
            for neuron_id in neuron_range:
                # add spikes to list for printing
                spike_times[0].append(np.array(spikes.multiplexed[1]).round(5).tolist())
                header_spikes.append(f"{self.id_to_letter(alphabet_id)}[{neuron_id}]")

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

    def id_to_letter(self, id):
        return list(self.ALPHABET.keys())[id]


class SHTMTotal(SHTMBase, ABC):
    def __init__(self, log_permanence=None, log_weights=None, w_exc_inh_dyn=None, plasticity_cls=None, **kwargs):
        super().__init__(**kwargs)

        self.con_plastic = None
        self.w_exc_inh_dyn = w_exc_inh_dyn

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
            post_somas = PopulationView(self.get_neurons(NeuronType.Soma, symbol_id=self.ALPHABET[symbol_post]),
                                        list(range(self.p.Network.num_neurons)))
            if self.p.Synapses.dyn_inh_weights:
                proj_post_soma_inh = self.exc_to_inh[self.ALPHABET[symbol_post]]
            else:
                proj_post_soma_inh = None

            self.con_plastic.append(self.plasticity_cls(self.exc_to_exc[i_plastic], post_somas=post_somas, shtm=self,
                                                       proj_post_soma_inh=proj_post_soma_inh,
                                                       debug=debug, **self.p.Plasticity.dict()))

        for i_perm in self.log_permanence:
            self.con_plastic[i_perm].enable_permanence_logging()
        for i_perm in self.log_weights:
            self.con_plastic[i_perm].enable_weights_logging()

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

    def get_spike_times(self, runtime, dt):
        times = np.linspace(0., runtime, int(runtime / dt))

        spike_times_dendrite = np.zeros((self.p.Network.num_symbols, self.p.Network.num_neurons, len(times)),
                                        dtype=np.int8)
        spike_times_soma = np.zeros((self.p.Network.num_symbols, self.p.Network.num_neurons, len(times)), dtype=np.int8)

        for symbol_i in range(self.p.Network.num_symbols):
            for i_dendrite, dendrite_spikes in enumerate(self.get_neuron_data(NeuronType.Dendrite, symbol_id=symbol_i,
                                                                              value_type=RecTypes.SPIKES)):
                for spike_time in dendrite_spikes:
                    spike_id = int(spike_time / times[1])
                    spike_times_dendrite[symbol_i, i_dendrite, spike_id] = 1

            for i_soma, soma_spikes in enumerate(self.get_neuron_data(NeuronType.Soma, symbol_id=symbol_i,
                                                                      value_type=RecTypes.SPIKES)):
                for spike_time in soma_spikes:
                    spike_id = int(spike_time / times[1])
                    spike_times_soma[symbol_i, i_soma, spike_id] = 1

        return spike_times_dendrite, spike_times_soma

    def set_weights_exc_exc(self, new_weight, con_id, post_ids=None, p_con=1.0):
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

    def run(self, runtime=None, steps=200, plasticity_enabled=True, dyn_exc_inh=False):
        if runtime is None:
            runtime = self.p.Experiment.runtime

        if type(runtime) is str:
            if str(runtime).lower() == 'max':
                runtime = self.last_ext_spike_time + 0.1
        elif type(runtime) is float or type(runtime) is int:
            pass
        else:
            log.error("Error! Wrong runtime")

        for t in range(steps):
            log.info(f'Running emulation step {t + 1}/{steps}')

            # reset the simulator and the network state if not first run
            if pynn.get_current_time() > 0 and t > 0:
                self.reset()

            sim_start_time = pynn.get_current_time()
            log.info(f"Current time: {sim_start_time}")

            pynn.run(runtime)

            active_synapse_post = np.zeros((self.p.Network.num_symbols, self.p.Network.num_neurons))

            if plasticity_enabled:
                log.info("Starting plasticity calculations")
                # Prepare spike time matrices
                self.spike_times_dendrite, self.spike_times_soma = self.get_spike_times(runtime, self.p.Plasticity.dt)

                # Calculate plasticity for each synapse
                for i_plasticity, plasticity in enumerate(self.con_plastic):
                    plasticity(runtime, self.spike_times_dendrite, self.spike_times_soma, sim_start_time=sim_start_time)
                    log.info(f"Finished plasticity calculation {i_plasticity + 1}/{len(self.con_plastic)}")

                    if dyn_exc_inh:
                        w = self.exc_to_exc[i_plasticity].get("weight", format="array")
                        letter_id = self.ALPHABET[plasticity.get_post_symbol()]
                        active_synapse_post[letter_id, :] = np.logical_or(active_synapse_post[letter_id, :],
                                                                          np.any(w > 0, axis=0))

                if dyn_exc_inh and self.w_exc_inh_dyn is not None:
                    for i_inh in range(self.p.Network.num_symbols):
                        w = self.exc_to_inh.get("weight", format="array")
                        w[active_synapse_post[i_inh, :]] = self.w_exc_inh_dyn


class Plasticity(ABC):
    def __init__(self, projection: Projection, post_somas, shtm, proj_post_soma_inh=None, debug=False,
                 learning_factor=None, permanence_init_min=None, permanence_init_max=None, permanence_max=None, threshold=None, w_mature=None, y=None, lambda_plus=None,
                 lambda_minus=None, lambda_h=None, target_rate_h=None, tau_plus=None, tau_h=None, delta_t_min=None,
                 delta_t_max=None, dt=None, **kwargs):
        self.projection = projection
        self.proj_post_soma_inh = proj_post_soma_inh

        self.permanence_min = np.asarray(np.random.randint(0, 8, size=(len(self.projection),)), dtype=float)
        self.permanence = copy.copy(self.permanence_min)
        self.permanences = None
        self.weights = None
        self.shtm: SHTMTotal = shtm
        self.post_somas = post_somas
        self.debug = debug

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

        self.x = np.zeros((len(self.projection.pre)))
        self.z = np.zeros((len(self.projection.post)))

        self.symbol_id_pre = SHTMBase.ALPHABET[symbol_from_label(self.projection.label, ID_PRE)]
        self.symbol_id_post = SHTMBase.ALPHABET[symbol_from_label(self.projection.label, ID_POST)]

    def rule(self, permanence, threshold, x, z, runtime, permanence_min,
             neuron_spikes_pre, neuron_spikes_post_dendrite, neuron_spikes_post_soma, spike_times_dendrite,
             spike_times_soma, id_pre, id_post, sim_start_time=0.0):
        mature = False
        for i, t in enumerate(np.linspace(sim_start_time, sim_start_time+runtime, int(runtime / self.dt))):

            # True - if any pre-synaptic neuron spiked
            has_pre_spike = spike_times_soma[self.symbol_id_pre, id_pre, i]
            # True - if any post dendrite spiked
            has_post_dendritic_spike = spike_times_dendrite[self.symbol_id_post, id_post, i]

            if spike_times_soma[self.symbol_id_post, id_post, i] > 0:
                # Indicator function (1st step) - Number of presynaptic spikes within learning time window
                # for each postsynaptic spike
                I = [sum(
                    self.delta_t_min < (spike_post - spike_pre) < self.delta_t_max for spike_pre in neuron_spikes_pre)
                    for spike_post in neuron_spikes_post_soma]
                # Indicator function (2nd step) - Number of pairs of pre-/postsynaptic spikes
                # for which synapses are potentiated
                has_post_somatic_spike_I = sum(
                    (t <= spike < t + self.dt) and I[n] for n, spike in enumerate(neuron_spikes_post_soma))
            else:
                has_post_somatic_spike_I = 0

            # Spike trace of presynaptic neuron
            x += (- x / self.tau_plus) * self.dt + has_pre_spike
            # Spike trace of postsynaptic neuron based on daps
            z += (- z / self.tau_h) * self.dt + has_post_dendritic_spike

            dp_a = x * has_post_somatic_spike_I
            dp_b = self.y * has_pre_spike
            dp_c = (self.target_rate_h - z) * has_post_somatic_spike_I

            delta_permanence = (
                    (self.lambda_plus * dp_a
                     - self.lambda_minus * dp_b
                     + self.lambda_h * dp_c)
                    * self.permanence_max * self.dt)

            permanence += delta_permanence

            if delta_permanence != 0:
                if self.debug:
                    log.debug(
                        f"t: {round(t, 5)},  p: {round(permanence, 5)},  dp: {round(delta_permanence, 5)},  x: {round(x, 2)},"
                        f"z: {round(z, 2)}, dp_a: {round(dp_a, 3)}, dp_b: {round(dp_b, 3)}, dp_c: {round(dp_c, 3)}")

            permanence = np.clip(permanence, a_min=permanence_min, a_max=self.permanence_max)

        if permanence >= threshold:
            mature = True

        return permanence, x, z, mature

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

    def __call__(self, runtime: float, spike_times_dendrite, spike_times_soma, sim_start_time=0.0):
        if isinstance(self.projection.pre.celltype, SpikeSourceArray):
            spikes_pre = self.projection.pre.get("spike_times").value
            spikes_pre = np.array(spikes_pre)
            if spikes_pre.ndim == 1:
                spikes_pre = np.array([spikes_pre] * len(self.projection.pre))
        else:
            # spikes_pre = self.projection.pre.get_data("spikes").segments[-1].spiketrains
            spikes_pre = self.shtm.get_neuron_data(NeuronType.Soma, neurons=self.projection.pre,
                                                   value_type=RecTypes.SPIKES)
        # spikes_post_dendrite = self.projection.post.get_data("spikes").segments[-1].spiketrains
        spikes_post_dendrite = self.shtm.get_neuron_data(NeuronType.Dendrite, neurons=self.projection.post,
                                                         value_type=RecTypes.SPIKES)
        # spikes_post_somas = self.post_somas.get_data("spikes").segments[-1].spiketrains
        spikes_post_somas = self.shtm.get_neuron_data(NeuronType.Soma, neurons=self.post_somas,
                                                         value_type=RecTypes.SPIKES)
        weight = self.projection.get("weight", format="array")

        for c, connection in enumerate(self.get_connections()):
            i = self.get_connection_id_post(connection)
            j = self.get_connection_id_pre(connection)
            neuron_spikes_pre = spikes_pre[j]
            neuron_spikes_post_dendrite = np.array(spikes_post_dendrite[i])
            neuron_spikes_post_soma = spikes_post_somas[i]

            if self.debug:
                log.debug(f"Permanence calculation for connection {c} [{i}, {j}]")
                log.debug(f"Spikes pre [soma]: {neuron_spikes_pre}")
                log.debug(f"Spikes post [dend]: {neuron_spikes_post_dendrite}")
                log.debug(f"Spikes post [soma]: {neuron_spikes_post_soma}")

            permanence, x, z, mature = self.rule(permanence=self.permanence[c], threshold=self.threshold[c],
                                                 runtime=runtime, x=self.x[j], z=self.z[i],
                                                 permanence_min=self.permanence_min[c],
                                                 neuron_spikes_pre=neuron_spikes_pre,
                                                 neuron_spikes_post_dendrite=neuron_spikes_post_dendrite,
                                                 neuron_spikes_post_soma=neuron_spikes_post_soma,
                                                 spike_times_dendrite=spike_times_dendrite,
                                                 spike_times_soma=spike_times_soma, id_pre=j, id_post=i,
                                                 sim_start_time=sim_start_time)
            self.permanence[c] = permanence
            self.x[j] = x
            self.z[i] = z

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
            self.weights.append(np.copy(np.round(self.projection.get("weight", format="array").flatten(), 6)))

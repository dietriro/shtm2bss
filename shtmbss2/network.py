import numpy as np
import quantities
import copy
import pickle

from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
from pyNN.random import NumpyRNG
from tabulate import tabulate
from abc import ABC, abstractmethod

from pynn_brainscales import brainscales2 as pynn
from pynn_brainscales.brainscales2.standardmodels.cells import SpikeSourceArray
from pynn_brainscales.brainscales2.standardmodels.synapses import StaticSynapse

from shtmbss2.plot import plot_membrane
from shtmbss2.core.logging import log
from shtmbss2.core.parameters import Parameters
from shtmbss2.plot import plot_dendritic_events

ID_DENDRITE = 0
ID_SOMA = 1

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

        # Status variables
        self.runtime = None

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

    def load_params(self, **kwargs):
        self.p = Parameters(network_type=self, custom_params=kwargs)

    def init_network(self):
        self.init_neurons()
        self.init_connections()
        self.init_external_input()
        self.init_rec_exc()

    def init_neurons(self):
        self.neurons_exc = []
        for i in range(self.p.Network.num_symbols):
            dendrites, somas = self.init_neurons_exc()
            dendrites.record(['spikes'])
            somas.record(['spikes'])
            self.neurons_exc.append((dendrites, somas))

        self.neurons_inh = self.init_neurons_inh()

        self.neurons_ext = pynn.Population(self.p.Network.num_symbols, SpikeSourceArray())

        log.inf("Starting preprocessing/calibration...")
        pynn.preprocess()

        for dendrites, somas in self.neurons_exc:
            self.init_neurons_exc_post_preprocess(dendrites, somas)

        if self.p.Experiment.run_add_calib:
            self.run_add_calibration()

    def init_neurons_exc(self, num_neurons=None):
        if num_neurons is None:
            num_neurons = self.p.Network.num_neurons

        # TODO: remove once pynn_brainscales supports float values directly (bug currently)
        # pynn.cells.CalibHXNeuronCuba.default_parameters.update({"tau_syn_I": 2.})

        all_neurons = pynn.Population(num_neurons * 2, pynn.cells.CalibHXNeuronCuba(
            tau_m=self.p.Neurons.Excitatory.tau_m,
            tau_syn_I=self.p.Neurons.Excitatory.tau_syn_I,
            tau_syn_E=self.p.Neurons.Excitatory.tau_syn_E,
            v_rest=self.p.Neurons.Excitatory.v_rest,
            v_reset=self.p.Neurons.Excitatory.v_reset,
            v_thresh=self.p.Neurons.Excitatory.v_thresh,
            tau_refrac=self.p.Neurons.Excitatory.tau_refrac,
        ))

        dendrites = pynn.PopulationView(all_neurons, slice(ID_DENDRITE, num_neurons * 2, 2))
        somas = pynn.PopulationView(all_neurons, slice(ID_SOMA, num_neurons * 2, 2))

        somas.record(["spikes"])

        return dendrites, somas

    @staticmethod
    def init_neurons_exc_post_preprocess(dendrites, somas):
        for i in range(len(dendrites)):
            dendrites.actual_hwparams[i].multicompartment.enable_conductance = True
            dendrites.actual_hwparams[i].multicompartment.i_bias_nmda = 120
            dendrites.actual_hwparams[i].multicompartment.connect_soma_right = True
            dendrites.actual_hwparams[i].refractory_period.reset_holdoff = 0

        for i in range(len(somas)):
            somas.actual_hwparams[i].multicompartment.connect_soma = True

    def init_neurons_inh(self, num_neurons=None):
        if num_neurons is None:
            num_neurons = self.p.Network.num_symbols

        pynn.cells.CalibHXNeuronCuba.default_parameters.update({"tau_syn_E": 2.})

        pop = pynn.Population(num_neurons, pynn.cells.CalibHXNeuronCuba(
            tau_m=self.p.Neurons.Inhibitory.tau_m,
            tau_syn_I=self.p.Neurons.Inhibitory.tau_syn_I,
            tau_syn_E=self.p.Neurons.Inhibitory.tau_syn_E,
            v_rest=self.p.Neurons.Inhibitory.v_rest,
            v_reset=self.p.Neurons.Inhibitory.v_reset,
            v_thresh=self.p.Neurons.Inhibitory.v_thresh,
            tau_refrac=self.p.Neurons.Inhibitory.tau_refrac,
        ))

        pop.record(["spikes"])

        return pop

    @abstractmethod
    def init_external_input(self):
        pass

    def init_connections(self):
        self.ext_to_exc = []
        for i in range(self.p.Network.num_symbols):
            self.ext_to_exc.append(pynn.Projection(
                pynn.PopulationView(self.neurons_ext, [i]),
                self.neurons_exc[i][ID_SOMA],
                pynn.AllToAllConnector(),
                synapse_type=StaticSynapse(weight=self.p.Synapses.w_ext_exc),
                receptor_type="excitatory"))

        self.exc_to_exc = []
        num_connections = int(self.p.Network.num_neurons * self.p.Synapses.p_exc_exc)
        for i in range(self.p.Network.num_symbols):
            for j in range(self.p.Network.num_symbols):
                if i == j:
                    continue
                self.exc_to_exc.append(pynn.Projection(
                    self.neurons_exc[i][ID_SOMA],
                    self.neurons_exc[j][ID_DENDRITE],
                    pynn.FixedNumberPreConnector(num_connections,
                                                 rng=NumpyRNG(seed=j + i * self.p.Network.num_symbols)),
                    synapse_type=StaticSynapse(weight=self.p.Synapses.w_exc_exc),
                    receptor_type="excitatory",
                    label=f"exc-exc_{self.id_to_letter(i)}>{self.id_to_letter(j)}"))

        self.exc_to_inh = []
        for i in range(self.p.Network.num_symbols):
            self.exc_to_inh.append(pynn.Projection(
                self.neurons_exc[i][ID_SOMA],
                pynn.PopulationView(self.neurons_inh, [i]),
                pynn.AllToAllConnector(),
                synapse_type=StaticSynapse(weight=self.p.Synapses.w_exc_inh),
                receptor_type="excitatory"))

        self.inh_to_exc = []
        for i in range(self.p.Network.num_symbols):
            self.inh_to_exc.append(pynn.Projection(
                pynn.PopulationView(self.neurons_inh, [i]),
                self.neurons_exc[i][ID_SOMA],
                pynn.AllToAllConnector(),
                synapse_type=StaticSynapse(weight=self.p.Synapses.w_inh_exc),
                receptor_type="inhibitory"))

    def init_rec_exc(self, alphabet_id=1, neuron_id=1, neuron_type=1):
        # ToDo: What exactly are we recording here? External or excitatory?
        self.rec_neurons_exc = pynn.PopulationView(self.neurons_exc[alphabet_id][neuron_type], [neuron_id])
        self.rec_neurons_exc.record(["v", "spikes"])

    def reset_rec_exc(self):
        self.rec_neurons_exc.record(None)

    def plot_events(self, neuron_types="all", symbols="all", size=None, x_lim_lower=None, x_lim_upper=None, seq_start=0,
                    seq_end=None, fig_title="", file_path=None):
        if size is None:
            size = (12, 10)

        if type(neuron_types) is str and neuron_types == "all":
            neuron_types = [NeuronType.Dendrite, NeuronType.Soma, NeuronType.Inhibitory]
        elif type(neuron_types) is list:
            pass
        else:
            return

        if self.runtime is not None:
            max_time = self.runtime
        else:
            max_time = pynn.get_current_time()

        if x_lim_lower is None:
            x_lim_lower = 0.
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

        ax = None

        for i_symbol in symbols:
            if len(symbols) == 1:
                ax = axs
            else:
                ax = axs[i_symbol]

            neurons_all = dict()
            neurons_all[NeuronType.Dendrite], neurons_all[NeuronType.Soma], = self.neurons_exc[i_symbol]
            neurons_all[NeuronType.Inhibitory] = pynn.PopulationView(self.neurons_inh, [i_symbol])

            for neurons_i in neuron_types:
                # Retrieve and plot spikes from selected neurons
                spikes = [s.base for s in neurons_all[neurons_i].get_data("spikes").segments[-1].spiketrains]
                if neurons_i == NeuronType.Inhibitory:
                    spikes.append([])
                else:
                    spikes.insert(0, [])
                if neurons_i == NeuronType.Dendrite:
                    spikes_post = [s.base for s in
                                   neurons_all[NeuronType.Soma].get_data("spikes").segments[-1].spiketrains]
                    plot_dendritic_events(ax, spikes[1:], spikes_post, color=f"C{neurons_i.ID}",
                                          label=neurons_i.NAME.capitalize(), seq_start=seq_start, seq_end=seq_end)
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
            ax.grid(True, which='both', axis='both')

            # Generate y-tick-labels based on number of neurons per symbol
            y_tick_labels = ['Inh', '', '0'] + ['' for k in range(self.p.Network.num_neurons - 2)] + [
                str(self.p.Network.num_neurons - 1)]
            ax.set_yticklabels(y_tick_labels, rotation=45, fontsize=18)

        # Create custom legend for all plots
        custom_lines = [Line2D([0], [0], color=f"C{n.ID}", label=n.NAME.capitalize(), lw=3) for n in neuron_types]

        ax.set_xlabel("Time [ms]", fontsize=26, labelpad=14)
        ax.xaxis.set_ticks(np.arange(x_lim_lower, x_lim_upper, 0.02))
        ax.tick_params(axis='x', labelsize=18)

        plt.figlegend(handles=custom_lines, loc=(0.377, 0.885), ncol=3, labelspacing=0., fontsize=18, fancybox=True,
                      borderaxespad=4)

        fig.text(0.01, 0.5, "Symbol & Neuron ID", va="center", rotation="vertical", fontsize=26)

        fig.suptitle(fig_title, x=0.5, y=0.99, fontsize=26)

        if file_path is not None:
            plt.savefig(f"{file_path}.pdf")

            pickle.dump(fig, open(f'{file_path}.fig.pickle',
                                  'wb'))  # This is for Python 3 - py2 may need `file` instead of `open`

    def plot_v_exc(self, alphabet_range, neuron_range='all', size=None, neuron_type=ID_SOMA, runtime=0.1,
                   show_legend=False, file_path=None):
        self.reset_rec_exc()

        if size is None:
            size = (12, 10)

        if type(neuron_range) is str and neuron_range == 'all':
            neuron_range = range(self.p.Network.num_neurons)
        elif type(neuron_range) is list:
            pass
        else:
            return

        spike_times = [[]]
        header_spikes = list()

        fig, ax = plt.subplots(figsize=size)

        for alphabet_id in alphabet_range:
            for neuron_id in neuron_range:
                self.init_rec_exc(alphabet_id=alphabet_id, neuron_id=neuron_id, neuron_type=neuron_type)
                pynn.run(runtime)

                # Retrieve and save spike times
                spikes = self.rec_neurons_exc.get_data("spikes").segments[-1].spiketrains
                spike_times[0].append(np.array(spikes.multiplexed[1]).round(5).tolist())
                header_spikes.append(f"{self.id_to_letter(alphabet_id)}[{neuron_id}]")

                # plot_membrane(self.rec_neurons_exc, label=header_spikes[-1])

                membrane = self.rec_neurons_exc.get_data("v").segments[-1].irregularlysampledsignals[0]
                ax.plot(membrane.times, membrane, alpha=0.5, label=header_spikes[-1])

                self.reset_rec_exc()

        # ax.xaxis.set_ticks(np.arange(0.02, 0.06, 0.01))
        ax.tick_params(axis='x', labelsize=18)
        ax.tick_params(axis='y', labelsize=18)

        ax.set_xlabel("Time [ms]", labelpad=14, fontsize=26)
        ax.set_ylabel("Membrane Voltage [a.u.]", labelpad=14, fontsize=26)

        if show_legend:
            plt.legend()

            # Print spike times
        print(tabulate(spike_times, headers=header_spikes) + '\n')

        if file_path is not None:
            plt.savefig(f"{file_path}.pdf")

            pickle.dump(fig, open(f'{file_path}.fig.pickle',
                                  'wb'))  # This is for Python 3 - py2 may need `file` instead of `open`

    def id_to_letter(self, id):
        return list(self.ALPHABET.keys())[id]


class SHTMSingleNeuron(SHTMBase):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # Configuration
        self.wait_before_experiment = 0.003 * quantities.ms
        self.isi = 1 * quantities.us

        # External input
        self.pop_dendrite_in = None
        self.pop_soma_in = None

        # Connections
        self.proj_ref_in = None
        self.proj_soma_in = None
        self.proj_dendrite_in = None

    def init_network(self):
        self.init_neurons()
        self.init_external_input()
        self.init_connections()
        self.init_rec_exc(alphabet_id=0, neuron_id=0, neuron_type=0)

    def init_neurons(self):
        self.neurons_exc = []

        dendrite, soma, ref_neuron = self.init_neurons_exc()
        self.neurons_exc.append((dendrite, soma, ref_neuron))

        log.info("Starting preprocessing/calibration...")
        pynn.preprocess()

        for dendrites, somas, ref_neuron in self.neurons_exc:
            self.init_neurons_exc_post_preprocess(dendrites, somas)

    def init_neurons_exc(self, num_neurons=None):
        predictive_mode = True

        # TODO: remove once pynn_brainscales supports float values directly (bug currently)
        pynn.cells.CalibHXNeuronCuba.default_parameters.update({"tau_syn_I": 2.})

        pop_ref_neuron = pynn.Population(1, pynn.cells.CalibHXNeuronCuba(
            tau_m=5,
            tau_syn_I=0.5,
            tau_syn_E=5,
            v_rest=60,
            v_reset=60,
            v_thresh=120 if predictive_mode else 75,
            tau_refrac=2,
        ))

        ref_neuron = pynn.PopulationView(pop_ref_neuron, [0])
        dendrite, soma = super().init_neurons_exc(1)

        return dendrite, soma, ref_neuron

    def init_external_input(self):
        # Define input spikes, population and connections for dendritic coincidence spikes
        spikes_coincidence = self.wait_before_experiment + np.arange(10) * self.isi
        spikes_coincidence = np.array(spikes_coincidence.rescale(quantities.ms))
        self.pop_dendrite_in = pynn.Population(2,  # ToDo: Why 2 neurons?
                                               SpikeSourceArray(spike_times=spikes_coincidence))

        # Define input spikes, population and connections for somatic spikes
        self.pop_soma_in = pynn.Population(4,  # ToDo: Why 4 neurons?
                                           SpikeSourceArray(spike_times=[0.025]))

    def init_connections(self):
        dendrite, soma, ref_neuron = self.neurons_exc[0]

        self.proj_dendrite_in = pynn.Projection(self.pop_dendrite_in, dendrite,
                                                pynn.AllToAllConnector(),
                                                synapse_type=StaticSynapse(weight=63),
                                                receptor_type="excitatory")

        self.proj_soma_in = pynn.Projection(self.pop_soma_in, soma,
                                            pynn.AllToAllConnector(),
                                            synapse_type=StaticSynapse(weight=40),
                                            receptor_type="excitatory")

        self.proj_ref_in = pynn.Projection(self.pop_soma_in, ref_neuron,
                                           pynn.AllToAllConnector(),
                                           synapse_type=StaticSynapse(weight=40),
                                           receptor_type="excitatory")

    def plot_v_exc(self, alphabet_range=None, **kwargs):
        # Define neurons to be recorded
        recorded_neurons = [2, 0, 1]
        labels = ["point neuron", "dendrite", "soma"]
        fig, axs = plt.subplots(len(recorded_neurons), 1, sharex="all")
        spike_times = []
        num_spikes = 0

        ## Run network and plot results
        for idx, neuron_id in enumerate(recorded_neurons):
            # Define neuron to be recorded in this run
            self.init_rec_exc(alphabet_id=0, neuron_id=0, neuron_type=neuron_id)

            # Run simulation for 0.08 seconds
            pynn.run(0.08)

            # Retrieve voltage and spikes from recorded neuron
            mem_v = self.rec_neurons_exc.get_data("v").segments[-1].irregularlysampledsignals[0]
            spikes = self.rec_neurons_exc.get_data("spikes").segments[-1].spiketrains

            # Print spikes
            spike_times.append([labels[idx], ] + np.array(spikes.multiplexed[1]).round(5).tolist())
            num_spikes = max(num_spikes, len(spikes.multiplexed[1]))

            # Plot
            axs[idx].plot(mem_v.times, mem_v, alpha=0.5, label=labels[idx])
            axs[idx].legend()
            # axs[idx].set_ylim(100, 800)
            pynn.reset()
            self.reset_rec_exc()

        # Print spike times
        header_spikes = [f'Spike {i}' for i in range(num_spikes)]
        print(tabulate(spike_times, headers=['Neuron'] + header_spikes) + '\n')

        # Update figure, save and show it
        axs[-1].set_xlabel("time [ms]")
        fig.text(0.025, 0.5, "membrane potential [a.u.]", va="center", rotation="vertical")
        plt.savefig("../evaluation/excitatory_neuron.pdf")


class SHTMStatic(SHTMBase):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.last_ext_spike_time = None

    # def init_connections(self):
    #     super().init_connections()

    def init_external_input(self):
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


class SHTMPlasticity(SHTMSingleNeuron):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.weight = None
        self.permanences = None
        self.runtime = None
        self.vs = None
        self.plasticity = None

    def init_external_input(self):
        self.pop_dendrite_in = pynn.Population(8, pynn.cells.SpikeSourceArray(spike_times=[14e-3]))
        self.pop_soma_in = pynn.Population(1, pynn.cells.SpikeSourceArray(spike_times=[17e-3]))

    def init_connections(self):
        dendrite, soma, ref_neuron = self.neurons_exc[0]

        dendrite.record(['spikes'])
        soma.record(['spikes'])

        self.proj_dendrite_in = pynn.Projection(self.pop_dendrite_in, dendrite, pynn.AllToAllConnector(),
                                                pynn.standardmodels.synapses.StaticSynapse(weight=0),
                                                receptor_type="excitatory")
        self.proj_soma_in = pynn.Projection(self.pop_soma_in, soma, pynn.AllToAllConnector(),
                                            pynn.standardmodels.synapses.StaticSynapse(weight=200),
                                            receptor_type="excitatory")

        self.plasticity = PlasticitySingleNeuron(self.proj_dendrite_in, post_somas=soma)
        self.plasticity.debug = False

    def run(self, runtime=0.1):
        dendrite, soma, ref_neuron = self.neurons_exc[0]

        self.runtime = runtime

        dendrite_v = []
        soma_v = []

        self.vs = (dendrite_v, soma_v)
        self.permanences = []
        self.weight = []

        # Todo: Remove if no longer needed
        # dendrite_s = []
        # soma_s = []
        # spikes = [dendrite_s, soma_s]
        # x = []
        # z = []

        for idx, pop in enumerate([dendrite, soma]):
            pop.record(["v", "spikes"])
            for t in range(200):
                log.info(f'Running emulation step {t + 1}/200 for neuron {idx + 1}/2')

                pynn.run(self.runtime)

                self.plasticity(self.runtime)

                # Gather data for visualization of single neuron plasticity
                self.vs[idx].append(pop.get_data("v").segments[-1].irregularlysampledsignals[0])
                self.permanences.append(copy.copy(self.plasticity.permanence))
                self.weight.append(self.proj_dendrite_in.get("weight", format="array"))

                # Todo: Remove if no longer needed
                # spikes[idx].append(pop.get_data("spikes").segments[-1].spiketrains)
                # x.append(self.plasticity.x[0])
                # z.append(self.plasticity.z[0])

            pop.record(None)
            pop.record(["spikes"])  # Todo: Remove?! Unnecessary?!

    def test_plasticity(self):
        # Todo: still needed? Which plasticity initialization did we use?
        dendrite, soma, ref_neuron = self.neurons_exc[0]

        test_projection = pynn.Projection(pynn.PopulationView(self.pop_dendrite_in, [0]), dendrite,
                                          pynn.AllToAllConnector(),
                                          pynn.standardmodels.synapses.StaticSynapse(weight=0),
                                          receptor_type="excitatory")

        plasticity = PlasticitySingleNeuron(test_projection, post_somas=soma)

        plasticity.debug = False

        permanence = 0.
        x = 0.
        z = 0.
        runtime = 0.1

        neuron_spikes_pre = np.array([10. / 1e3])
        neuron_spikes_post = np.array([12. / 1e3])
        neuron_spikes_post_soma = np.array([15. / 1e3])

        for i in range(1):
            plasticity.rule(permanence, 20., x, z, runtime, neuron_spikes_pre, neuron_spikes_post,
                            neuron_spikes_post_soma)

    def plot_events_plasticity(self):
        fig, (ax_pre, ax_d, ax_s, ax_p, ax_w) = plt.subplots(5, 1, sharex=True)

        times_dendrite = np.concatenate([np.array(self.vs[0][i].times) + self.runtime * i for i in range(200)])
        vs_dendrite = np.concatenate([np.array(self.vs[0][i]) for i in range(200)])
        times_soma = np.concatenate([np.array(self.vs[1][i].times) + self.runtime * i for i in range(200)])
        vs_soma = np.concatenate([np.array(self.vs[1][i]) for i in range(200)])

        ax_pre.eventplot(
            np.concatenate(
                [np.array(self.pop_dendrite_in.get("spike_times").value) + i * self.runtime for i in range(200)]),
            label="pre", lw=.2)
        ax_pre.legend()
        ax_d.plot(times_dendrite, vs_dendrite, label="dendrite", lw=.2)
        ax_d.legend()
        ax_s.plot(times_soma, vs_soma, label="soma", lw=.2)
        ax_s.legend()
        ax_p.plot(np.linspace(0., self.runtime * 200., 200), self.permanences[:200], label="permanence")
        ax_p.set_ylabel("P [a.u.]")
        # ax_p.legend()
        ax_w.plot(np.linspace(0., self.runtime * 200., 200),
                  np.array([w for w in np.array(self.weight[:200]).squeeze().T if any(w)]).T, label="weight")
        ax_w.set_ylabel("w [a.u.]")
        # ax_w.legend()
        ax_w.set_xlabel("time [ms]")
        plt.savefig("../evaluation/plasticity.pdf")


class SHTMTotal(SHTMStatic):
    def __init__(self, log_permanence=None, log_weights=None, w_exc_inh_dyn=None, **kwargs):
        super().__init__(**kwargs)

        self.runtime = None
        self.con_plastic = None
        self.w_exc_inh_dyn = w_exc_inh_dyn

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
            letter_post = self.exc_to_exc[i_plastic].label.split('_')[1].split('>')[1]
            # Create population view of all post synaptic somas
            post_somas = pynn.PopulationView(self.neurons_exc[self.ALPHABET[letter_post]][ID_SOMA],
                                             list(range(self.p.Network.num_neurons)))
            if self.p.Synapses.dyn_inh_weights:
                proj_post_soma_inh = self.exc_to_inh[self.ALPHABET[letter_post]]
            else:
                proj_post_soma_inh = None

            self.con_plastic.append(Plasticity(self.exc_to_exc[i_plastic], post_somas=post_somas,
                                               proj_post_soma_inh=proj_post_soma_inh,
                                               mature_weight=self.p.Plasticity.w_mature,
                                               learning_factor=self.p.Plasticity.learning_factor,
                                               debug=debug))

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
            dendrites, somas = self.neurons_exc[symbol_i]

            for i_dendrite, dendrite_spikes in enumerate(dendrites.get_data("spikes").segments[-1].spiketrains):
                for spike_time in dendrite_spikes:
                    spike_id = int(spike_time / times[1])
                    spike_times_dendrite[symbol_i, i_dendrite, spike_id] = 1

            for i_soma, soma_spikes in enumerate(somas.get_data("spikes").segments[-1].spiketrains):
                for spike_time in soma_spikes:
                    spike_id = int(spike_time / times[1])
                    spike_times_soma[symbol_i, i_soma, spike_id] = 1

        return spike_times_dendrite, spike_times_soma

    def set_weights_exc_exc(self, new_weight, con_id, post_ids=None, p_con=1.0):
        weights = self.con_plastic[con_id].projection.get("weight", format="array")

        if post_ids is None:
            post_ids = range(weights.shape[1])

        for i in post_ids:
            pre_ids = weights[:, i] == 0
            pre_ids = pre_ids[:int(p_con * len(pre_ids))]
            weights[pre_ids, i] = new_weight

        self.con_plastic[con_id].projection.set(weight=weights)
        self.con_plastic[con_id].mature_weight = new_weight

        return self.con_plastic[con_id].projection.get("weight", format="array")

    def run(self, runtime=None, steps=200, plasticity_enabled=True, dyn_exc_inh=False):
        if runtime is None:
            runtime = self.p.Experiment.runtime

        if type(runtime) is str:
            if str(runtime).lower() == 'max':
                runtime = self.last_ext_spike_time + 0.1
        elif type(runtime) is float:
            pass
        else:
            log.error("Error! Wrong runtime")

        self.p.Experiment.runtime = runtime

        for t in range(steps):
            log.info(f'Running emulation step {t + 1}/{steps}')

            log.info(f"Current time: {pynn.get_current_time()}")
            pynn.run(self.runtime)

            active_synapse_post = np.zeros((self.p.Network.num_symbols, self.p.Network.num_neurons))

            if plasticity_enabled:
                log.info("Starting plasticity calculations")
                # Prepare spike time matrices
                self.spike_times_dendrite, self.spike_times_soma = self.get_spike_times(runtime, 0.1e-3)

                # Calculate plasticity for each synapse
                for i_plasticity, plasticity in enumerate(self.con_plastic):
                    plasticity(self.runtime, self.spike_times_dendrite, self.spike_times_soma)
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


class Plasticity:
    def __init__(self, projection: pynn.Projection, post_somas=None, proj_post_soma_inh=None,
                 mature_weight=63, learning_factor=1, debug=False):
        self.projection = projection
        self.proj_post_soma_inh = proj_post_soma_inh
        # print("inside constructor")

        self.permanence_min = np.asarray(np.random.randint(0, 8, size=(len(self.projection),)), dtype=float)
        self.permanence = copy.copy(self.permanence_min)
        self.permanences = None
        self.weights = None
        self.permanence_max = 20.
        self.threshold = np.ones((len(self.projection))) * 10.
        self.lambda_plus = 0.08e3 * learning_factor
        self.lambda_minus = 0.0015e3 * learning_factor
        self.lambda_h = 0.014e3 * learning_factor
        self.tau_plus = 20e-3
        self.tau_h = 440e-3
        self.target_rate_h = 1.
        self.y = 1.
        self.delta_t_min = 4e-3
        self.delta_t_max = 80e-3
        self.dt = 0.1e-3
        self.post_somas = post_somas
        self.mature_weight = mature_weight
        self.debug = debug
        # print(self.delta_t_min, self.delta_t_max)

        self.x = np.zeros((len(self.projection.pre)))
        self.z = np.zeros((len(self.projection.post)))

        self.symbol_id_pre = SHTMBase.ALPHABET[symbol_from_label(self.projection.label, ID_PRE)]
        self.symbol_id_post = SHTMBase.ALPHABET[symbol_from_label(self.projection.label, ID_POST)]

    def rule(self, permanence, threshold, x, z, runtime, permanence_min,
             neuron_spikes_pre, neuron_spikes_post_dendrite, neuron_spikes_post_soma, spike_times_dendrite,
             spike_times_soma, id_pre, id_post):
        mature = False
        for i, t in enumerate(np.linspace(0., runtime, int(runtime / self.dt))):

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
        connection_ids = (f"{self.projection.connections[0].presynaptic_index}>"
                          f"{self.projection.connections[0].postsynaptic_index}")
        return connection_ids

    def get_all_connection_ids(self):
        connection_ids = []
        for con in self.projection.connections:
            connection_ids.append(f"{con.presynaptic_index}>{con.postsynaptic_index}")
        return connection_ids

    def __call__(self, runtime: float, spike_times_dendrite, spike_times_soma):
        if isinstance(self.projection.pre.celltype, pynn.cells.SpikeSourceArray):
            spikes_pre = self.projection.pre.get("spike_times").value
            spikes_pre = np.array(spikes_pre)
            if spikes_pre.ndim == 1:
                spikes_pre = np.array([spikes_pre] * len(self.projection.pre))
        else:
            spikes_pre = self.projection.pre.get_data("spikes").segments[-1].spiketrains
        spikes_post_dendrite = self.projection.post.get_data("spikes").segments[-1].spiketrains
        spikes_post_somas = self.post_somas.get_data("spikes").segments[-1].spiketrains
        weight = self.projection.get("weight", format="array")

        for c, connection in enumerate(self.projection.connections):
            i = connection.postsynaptic_index
            j = connection.presynaptic_index
            neuron_spikes_pre = spikes_pre[j]
            neuron_spikes_post_dendrite = np.array(spikes_post_dendrite[i])
            neuron_spikes_post_soma = spikes_post_somas[i]

            if self.debug:
                log.debug(f"Permanence calculation for connection {c} [{i}, {j}]")

            permanence, x, z, mature = self.rule(permanence=self.permanence[c], threshold=self.threshold[c],
                                                 runtime=runtime, x=self.x[j], z=self.z[i],
                                                 permanence_min=self.permanence_min[c],
                                                 neuron_spikes_pre=neuron_spikes_pre,
                                                 neuron_spikes_post_dendrite=neuron_spikes_post_dendrite,
                                                 neuron_spikes_post_soma=neuron_spikes_post_soma,
                                                 spike_times_dendrite=spike_times_dendrite,
                                                 spike_times_soma=spike_times_soma, id_pre=j, id_post=i)
            self.permanence[c] = permanence
            self.x[j] = x
            self.z[i] = z

            if mature:
                weight[j, i] = self.mature_weight
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


class PlasticitySingleNeuron:
    def __init__(self, projection: pynn.Projection, post_somas: pynn.PopulationView):
        self.projection = projection
        log.debug("inside constructor")

        self.permanence_min = np.asarray(np.random.randint(0, 8, size=(len(self.projection),)), dtype=float)
        self.permanence = copy.copy(self.permanence_min)
        self.permanence_max = 20.
        self.threshold = np.ones((len(self.projection))) * 20.
        self.lambda_plus = 0.08 * 1e3
        self.tau_plus = 20. / 1e3
        self.lambda_minus = 0.0015 * 1e3
        self.target_rate_h = 1.
        self.lambda_h = 0.014 * 1e3
        self.tau_h = 440. / 1e3
        self.y = 1.
        self.delta_t_min = 4e-3
        self.delta_t_max = 8e-3
        self.dt = 0.1e-3
        self.post_somas = post_somas
        self.mature_weight = 63
        self.debug = False

        self.x = np.zeros((len(self.projection.pre)))
        self.z = np.zeros((len(self.projection.post)))

    def rule(self, permanence, threshold, x, z, runtime, neuron_spikes_pre, neuron_spikes_post_dendrite,
             neuron_spikes_post_soma):
        mature = False
        for t in np.linspace(0., runtime, int(runtime / self.dt)):
            if self.debug:
                log.debug(t, round(permanence, 5), round(x, 2), round(z, 2))

            # True - if any pre-synaptic neuron spiked
            has_pre_spike = any(t <= spike < t + self.dt for spike in neuron_spikes_pre)
            # True - if any post
            has_post_dendritic_spike = any(t <= spike < t + self.dt for spike in neuron_spikes_post_dendrite)

            # Indicator function (1st step) - Number of presynaptic spikes within learning time window
            # for each postsynaptic spike
            I = [sum(self.delta_t_min < (spike_post - spike_pre) < self.delta_t_max for spike_pre in neuron_spikes_pre)
                 for spike_post in neuron_spikes_post_soma]
            # Indicator function (2nd step) - Number of pairs of pre-/postsynaptic spikes
            # for which synapses are potentiated
            has_post_somatic_spike_I = sum(
                (t <= spike < t + self.dt) and I[n] for n, spike in enumerate(neuron_spikes_post_soma))

            # Spike trace of presynaptic neuron
            x += (- x / self.tau_plus) * self.dt + has_pre_spike
            # Spike trace of postsynaptic neuron based on daps
            z += (- z / self.tau_h) * self.dt + has_post_dendritic_spike

            permanence += (self.lambda_plus * x * has_post_somatic_spike_I
                           - self.lambda_minus * self.y * has_pre_spike
                           + self.lambda_h * (
                                   self.target_rate_h - z) * has_post_somatic_spike_I) * self.permanence_max * self.dt
            if permanence >= threshold:
                mature = True
        return permanence, x, z, mature

    def __call__(self, runtime: float):
        if isinstance(self.projection.pre.celltype, pynn.cells.SpikeSourceArray):
            spikes_pre = self.projection.pre.get("spike_times").value
            spikes_pre = np.array(spikes_pre)
            if spikes_pre.ndim == 1:
                spikes_pre = np.array([spikes_pre] * len(self.projection.pre))
        else:
            spikes_pre = self.projection.pre.get_data("spikes").segments[-1].spiketrains

        spikes_post_dentrite = self.projection.post.get_data("spikes").segments[-1].spiketrains
        spikes_post_somas = self.post_somas.get_data("spikes").segments[-1].spiketrains

        weight = self.projection.get("weight", format="array")

        for c, connection in enumerate(self.projection.connections):
            i = connection.postsynaptic_index
            j = connection.presynaptic_index
            neuron_spikes_pre = spikes_pre[j]
            neuron_spikes_post_dendrite = np.array(spikes_post_dentrite[i])
            neuron_spikes_post_soma = np.array(spikes_post_somas[i])

            permanence, x, z, mature = self.rule(permanence=self.permanence[c], threshold=self.threshold[c],
                                                 runtime=runtime, x=self.x[j], z=self.z[i],
                                                 neuron_spikes_pre=neuron_spikes_pre,
                                                 neuron_spikes_post_dendrite=neuron_spikes_post_dendrite,
                                                 neuron_spikes_post_soma=neuron_spikes_post_soma)
            self.permanence[c] = permanence
            self.x[j] = x
            self.z[i] = z

            if weight[c] != self.mature_weight and mature:
                weight[c] = self.mature_weight
        self.projection.set(weight=weight)

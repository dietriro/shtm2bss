import numpy as np
import quantities
import copy

from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
from tabulate import tabulate
from abc import ABC, abstractmethod

from pynn_brainscales import brainscales2 as pynn
from pynn_brainscales.brainscales2.standardmodels.cells import SpikeSourceArray
from pynn_brainscales.brainscales2.standardmodels.synapses import StaticSynapse

from shtmbss2.plot import plot_membrane

ID_DENDRITE = 0
ID_SOMA = 1


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

    def __init__(self, alphabet_size=14, num_neurons_per_symbol=6):
        # Initialize parameters
        self.alphabet_size = alphabet_size
        self.num_neurons_per_symbol = num_neurons_per_symbol
        self.delta_stimulus = 40e-3
        self.delta_sequence = 100e-3

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

    def init_network(self):
        self.init_neurons(v_threshold)
        self.init_connections()
        self.init_external_input()
        self.init_rec_exc()

    def init_neurons(self):
        self.neurons_exc = []
        for i in range(self.alphabet_size):
            dendrites, somas = self.init_neurons_exc()
            dendrites.record(['spikes'])
            somas.record(['spikes'])
            self.neurons_exc.append((dendrites, somas))

        self.neurons_inh = self.init_neurons_inh()

        self.neurons_ext = pynn.Population(self.alphabet_size, SpikeSourceArray())

        pynn.preprocess()

        for dendrites, somas in self.neurons_exc:
            self.init_neurons_exc_post_preprocess(dendrites, somas)

    def init_neurons_exc(self, num_neurons=None):
        if num_neurons is None:
            num_neurons = self.num_neurons_per_symbol

        predictive_mode = True

        # TODO: remove once pynn_brainscales supports float values directly (bug currently)
        pynn.cells.CalibHXNeuronCuba.default_parameters.update({"tau_syn_I": 2.}) 

        all_neurons = pynn.Population(num_neurons * 2, pynn.cells.CalibHXNeuronCuba(
            tau_m=10,
            tau_syn_I=[3, 1] * num_neurons,
            tau_syn_E=[2, 5] * num_neurons,
            v_rest=v_rest,
            v_reset=v_reset * num_neurons,
            v_thresh=v_thresh * num_neurons,
            tau_refrac=[60, 10] * num_neurons,
        ))

        dendrites = pynn.PopulationView(all_neurons, slice(0, num_neurons * 2, 2))
        somas = pynn.PopulationView(all_neurons, slice(1, num_neurons * 2, 2))

        somas.record(["spikes"])

        return dendrites, somas

    @staticmethod
    def init_neurons_exc_post_preprocess(dendrites, somas):
        for i in range(len(dendrites)):
            dendrites.actual_hwparams[i].multicompartment.enable_conductance = True
            dendrites.actual_hwparams[i].multicompartment.i_bias_nmda = 30
            dendrites.actual_hwparams[i].multicompartment.connect_soma_right = True
            dendrites.actual_hwparams[i].refractory_period.reset_holdoff = 0

        for i in range(len(somas)):
            somas.actual_hwparams[i].multicompartment.connect_soma = True

    def init_neurons_inh(self, num_neurons=None):
        if num_neurons is None:
            num_neurons = self.alphabet_size

        pop = pynn.Population(num_neurons, pynn.cells.CalibHXNeuronCuba(
        #    cm=63,  # [0, 63]
            tau_m=5,
            tau_syn_I=10,
            tau_syn_E=0.5,
        #    v_rest=80,  # CADC lsb
        #    v_reset=80,  # CADC lsb
        #    v_thresh=100,  # CADC lsb
        #    i_synin_gm_I=500,  # capmem current lsb
        #    i_synin_gm_E=500,  # capmem current lsb
            tau_refrac=2,
        ))

        pop.record(["spikes"])

        return pop

    @abstractmethod
    def init_external_input(self):
        pass

    def init_connections(self, w_ext_exc=200, w_exc_exc=0.01, w_exc_inh=60, w_inh_exc=-80, p_exc_exc=0.2):
        self.ext_to_exc = []
        for i in range(self.alphabet_size):
            self.ext_to_exc.append(pynn.Projection(
                pynn.PopulationView(self.neurons_ext, [i]),
                self.neurons_exc[i][ID_SOMA],
                pynn.AllToAllConnector(),
                synapse_type=StaticSynapse(weight=w_ext_exc),
                receptor_type="excitatory"))

        self.exc_to_exc = []
        for i in range(self.alphabet_size):
            for j in range(self.alphabet_size):
                if i == j:
                    continue
                self.exc_to_exc.append(pynn.Projection(
                    self.neurons_exc[i][ID_SOMA],
                    self.neurons_exc[j][ID_DENDRITE],
                    pynn.FixedProbabilityConnector(p_exc_exc),  # Todo: Maybe replace this with FixedNumberPreConnector
                    synapse_type=StaticSynapse(weight=w_exc_exc),
                    receptor_type="excitatory",
                    label=f"exc-exc_{self.id_to_letter(i)}>{self.id_to_letter(j)}"))

        self.exc_to_inh = []
        for i in range(self.alphabet_size):
            self.exc_to_inh.append(pynn.Projection(
                self.neurons_exc[i][ID_SOMA],
                pynn.PopulationView(self.neurons_inh, [i]),
                pynn.AllToAllConnector(),
                synapse_type=StaticSynapse(weight=w_exc_inh),
                receptor_type="excitatory"))

        self.inh_to_exc = []
        for i in range(self.alphabet_size):
            self.inh_to_exc.append(pynn.Projection(
                pynn.PopulationView(self.neurons_inh, [i]),
                self.neurons_exc[i][ID_SOMA],
                pynn.AllToAllConnector(),
                synapse_type=StaticSynapse(weight=w_inh_exc),
                receptor_type="inhibitory"))

    def init_rec_exc(self, alphabet_id=1, neuron_id=1, neuron_type=1):
        # ToDo: What exactly are we recording here? External or excitatory?
        self.rec_neurons_exc = pynn.PopulationView(self.neurons_exc[alphabet_id][neuron_type], [neuron_id])
        self.rec_neurons_exc.record(["v", "spikes"])

    def reset_rec_exc(self):
        self.rec_neurons_exc.record(None)

    def plot_events(self, neuron_types="all"):
        fig, axs = plt.subplots(self.alphabet_size, 1, sharex="all", figsize=(12, 10))

        for i in range(self.alphabet_size):
            neurons_all = dict()
            neurons_all[NeuronType.Dendrite], neurons_all[NeuronType.Soma], = self.neurons_exc[i]
            neurons_all[NeuronType.Inhibitory] = pynn.PopulationView(self.neurons_inh, [i])

            if type(neuron_types) is str and neuron_types == "all":
                neuron_types = [NeuronType.Dendrite, NeuronType.Soma, NeuronType.Inhibitory]
            elif type(neuron_types) is list:
                pass
            else:
                return

            for neurons_i in neuron_types:
                # Retrieve and plot spikes from selected neurons
                spikes = [s.base for s in neurons_all[neurons_i].get_data("spikes").segments[-1].spiketrains]
                if neurons_i == NeuronType.Inhibitory:
                    spikes.append([])
                else:
                    spikes.insert(0, [])
                axs[i].eventplot(spikes, label=neurons_i.NAME, color=f"C{neurons_i.ID}")

            # Configure the plot layout
            axs[i].set_xlim(0., pynn.get_current_time())
            axs[i].set_ylim(-1, self.num_neurons_per_symbol + 1)
            axs[i].yaxis.set_ticks(range(self.num_neurons_per_symbol + 1))
            axs[i].set_ylabel(self.id_to_letter(i), weight='bold')
            axs[i].grid(True, which='both', axis='both')

            # Generate y-tick-labels based on number of neurons per symbol
            y_tick_labels = ['Inh', '0'] + ['' for k in range(self.num_neurons_per_symbol - 2)] + [
                str(self.num_neurons_per_symbol - 1)]
            axs[i].set_yticklabels(y_tick_labels)

        # Create custom legend for all plots
        custom_lines = [Line2D([0], [0], color=f"C{n.ID}", label=n.NAME, lw=3) for n in neuron_types]

        plt.figlegend(handles=custom_lines, loc='upper center', ncol=3, labelspacing=0.)

        axs[-1].set_xlabel("Time [ms]")
        fig.text(0.02, 0.5, "Symbol", va="center", rotation="vertical")

    def plot_v_exc(self, alphabet_range):
        self.reset_rec_exc()

        for alphabet_id in alphabet_range:
            for neuron_id in range(self.num_neurons_per_symbol):
                self.init_rec_exc(alphabet_id=alphabet_id, neuron_id=neuron_id)
                pynn.run(0.1)
                plot_membrane(self.rec_neurons_exc)
                self.reset_rec_exc()

    def id_to_letter(self, id):
        return list(self.ALPHABET.keys())[id]


class SHTMSingleNeuron(SHTMBase):
    def __init__(self, alphabet_size, num_neurons_per_symbol):
        super().__init__(alphabet_size, num_neurons_per_symbol)

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

    def plot_v_exc(self, alphabet_range=None):
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
    def __init__(self, alphabet_size, num_neurons_per_symbol):
        super().__init__(alphabet_size=alphabet_size, num_neurons_per_symbol=num_neurons_per_symbol)

        self.last_ext_spike_time = None

    # def init_connections(self):
    #     super().init_connections()

    def init_external_input(self, sequence=None, num_repetitions=1):
        spike_times = [list() for i in range(self.alphabet_size)]
        spike_time = None

        if sequence is None:
            spike_times[0].append(0.04)
        else:
            sequence_offset = 0
            for i_rep in range(num_repetitions):
                for i_element, element in enumerate(sequence):
                    spike_time = sequence_offset + i_element * self.delta_sequence + self.delta_stimulus
                    spike_times[self.ALPHABET[element]].append(spike_time)
                sequence_offset = spike_time + self.delta_sequence

        self.last_ext_spike_time = spike_time

        print(f'Initialized external input for sequence {sequence}')
        print(f'Spike times:')
        for i_letter, letter_spikes in enumerate(spike_times):
            print(f'{list(self.ALPHABET.keys())[i_letter]}: {spike_times[i_letter]}')

        self.neurons_ext.set(spike_times=spike_times)


class SHTMPlasticity(SHTMSingleNeuron):
    def __init__(self, alphabet_size, num_neurons_per_symbol):
        super().__init__(alphabet_size=alphabet_size, num_neurons_per_symbol=num_neurons_per_symbol)

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
                print(f'Running emulation step {t + 1}/200 for neuron {idx + 1}/2')

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
    def __init__(self, alphabet_size, num_neurons_per_symbol, log_permanence=None, log_weights=None):
        super().__init__(alphabet_size=alphabet_size, num_neurons_per_symbol=num_neurons_per_symbol)

        self.runtime = None
        self.con_plastic = None

        if log_permanence is None:
            self.log_permanence = list()
        elif type(log_permanence) is str and log_permanence.lower() == "all":
            self.log_permanence = range(self.alphabet_size ** 2 - self.alphabet_size)
        else:
            self.log_permanence = log_permanence

        if log_weights is None:
            self.log_weights = list()
        elif type(log_weights) is str and log_weights.lower() == "all":
            self.log_weights = range(self.alphabet_size ** 2 - self.alphabet_size)
        else:
            self.log_weights = log_weights

    def init_neurons(self):
        super().init_neurons()

    def init_connections(self, debug=False, w_ext_exc=200, w_exc_exc=0.01, w_exc_inh=60, w_inh_exc=-80, p_exc_exc=0.2):
        super().init_connections(w_ext_exc, w_exc_exc, w_exc_inh, w_inh_exc, p_exc_exc)

        self.con_plastic = list()

        for i_plastic in range(len(self.exc_to_exc)):
            # Retrieve id (letter) of post synaptic neuron population
            letter_post = self.exc_to_exc[i_plastic].label.split('_')[1].split('>')[1]
            # Create population view of all post synaptic somas
            post_somas = pynn.PopulationView(self.neurons_exc[self.ALPHABET[letter_post]][ID_SOMA],
                                             list(range(self.num_neurons_per_symbol)))

            self.con_plastic.append(Plasticity(self.exc_to_exc[i_plastic], post_somas=post_somas, debug=debug))

        for i_perm in self.log_permanence:
            self.con_plastic[i_perm].enable_permanence_logging()
        for i_perm in self.log_weights:
            self.con_plastic[i_perm].enable_weights_logging()

    def print_permanence_dif(self):
        for i_perm in self.log_permanence:
            permanence_diff = self.con_plastic[i_perm].permanences[-1] - self.con_plastic[i_perm].permanences[0]
            print(
                f"Permanence diff for {self.con_plastic[i_perm].projection.label} ({i_perm}): {list(permanence_diff)}")

    def plot_permanence_diff(self):
        fig, axs = plt.subplots(len(self.log_permanence), 1, sharex="all", figsize=(10, 7))

        for i_perm in self.log_permanence:
            permanence_diff = self.con_plastic[i_perm].permanences[-1] - self.con_plastic[i_perm].permanences[0]
            num_connections = len(permanence_diff)

            colors = ['C0' if p >= 0 else 'C1' for p in permanence_diff]

            axs[i_perm].bar(range(num_connections), permanence_diff, color=colors)
            axs[i_perm].set_ylabel(self.con_plastic[i_perm].projection.label.split('_')[1], weight='bold')
            axs[i_perm].yaxis.set_ticks([0])
            axs[i_perm].xaxis.set_ticks(range(0, num_connections))
            axs[i_perm].grid(True, which='both', axis='both')

        axs[-1].set_xlabel("Connection [#]")
        fig.text(0.02, 0.5, "Permanence diff / connection direction", va="center", rotation="vertical")

    def plot_permanence(self):
        fig, axs = plt.subplots(len(self.log_permanence), 1, sharex="all", figsize=(30, 30))

        for i_perm in self.log_permanence:
            permanence = self.con_plastic[i_perm].permanences
            num_connections = len(permanence)

            axs[i_perm].plot(range(num_connections), permanence)
            axs[i_perm].set_ylabel(self.con_plastic[i_perm].projection.label.split('_')[1], weight='bold')
            axs[i_perm].grid(True, which='both', axis='both')

        axs[-1].set_xlabel("Connection [#]")
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

    def run(self, runtime=0.1, steps=200, plasticity_enabled=True):
        if type(runtime) is str:
            if str(runtime).lower() == 'max':
                runtime = self.last_ext_spike_time + 0.1
        elif type(runtime) is float:
            pass
        else:
            print("Error! Wrong runtime")

        self.runtime = runtime

        # dendrite_v = []
        # soma_v = []

        # self.vs = (dendrite_v, soma_v)
        # self.permanences = []
        # self.weight = []

        # Todo: Remove if no longer needed
        # dendrite_s = []
        # soma_s = []
        # spikes = [dendrite_s, soma_s]
        # x = []
        # z = []

        # pop.record(["v", "spikes"])
        # Todo: Why do we run multiple steps with 0.1 runtime? Is that 0.1 ms?
        for t in range(steps):
            print(f'Running emulation step {t + 1}/{steps}')

            print(f"Current time: {pynn.get_current_time()}")
            pynn.run(self.runtime)

            if plasticity_enabled:
                print("Starting plasticity calculations")
                for i_plasticity, plasticity in enumerate(self.con_plastic):
                    plasticity(self.runtime)
                    print(f"Finished plasticity calculation {i_plasticity+1}/{len(self.con_plastic)}")

            # Gather data for visualization of single neuron plasticity
            # self.vs[idx].append(pop.get_data("v").segments[-1].irregularlysampledsignals[0])
            # self.permanences.append(copy.copy(self.plasticity.permanence))
            # self.weight.append(self.proj_dendrite_in.get("weight", format="array"))

            # Todo: Remove if no longer needed
            # spikes[idx].append(pop.get_data("spikes").segments[-1].spiketrains)
            # x.append(self.plasticity.x[0])
            # z.append(self.plasticity.z[0])

        # pop.record(None)
        # pop.record(["spikes"])  # Todo: Remove?! Unnecessary?!


class Plasticity:
    def __init__(self, projection: pynn.Projection, post_somas=None, debug=False):
        self.projection = projection
        # print("inside constructor")

        self.permanence_min = np.asarray(np.random.randint(0, 8, size=(len(self.projection),)), dtype=float)
        self.permanence = copy.copy(self.permanence_min)
        self.permanences = None
        self.permanence_max = 20.
        self.weights = None
        self.threshold = np.ones((len(self.projection))) * 20.
        self.lambda_plus = 0.08
        self.tau_plus = 20. / 1e3
        self.lambda_minus = 0.0015
        self.target_rate_h = 1.
        self.lambda_h = 0.014
        self.tau_h = 440. / 1e3
        self.y = 1.
        self.delta_t_min = 4e-3
        self.delta_t_max = 80e-3
        self.dt = 0.1e-3
        self.post_somas = post_somas
        self.mature_weight = 63
        self.debug = debug
        # print(self.delta_t_min, self.delta_t_max)

        self.x = np.zeros((len(self.projection.pre)))
        self.z = np.zeros((len(self.projection.post)))

    def rule(self, permanence, threshold, x, z, runtime, permanence_min,
             neuron_spikes_pre, neuron_spikes_post_dendrite, neuron_spikes_post_soma):
        mature = False
        for t in np.linspace(0., runtime, int(runtime / self.dt)):

            # Todo: Improve efficiency in case no spikes occured
            # True - if any pre-synaptic neuron spiked
            has_pre_spike = any(t <= spike < t + self.dt for spike in neuron_spikes_pre)
            # True - if any post dendrite spiked
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
                    print(
                        f"t: {t},  p: {round(permanence, 5)},  dp: {delta_permanence},  x: {round(x, 2)},  "
                        f"z: {round(z, 2)}, dp_a: {round(dp_a, 3)}, dp_b: {round(dp_b, 3)}, dp_c: {round(dp_c, 3)}")
                    print(f"{neuron_spikes_pre}")

            if permanence >= threshold:
                mature = True

            # Todo: Enable again after debugging
            # permanence = np.clip(permanence, a_min=permanence_min, a_max=self.permanence_max)

        return permanence, x, z, mature

    def enable_permanence_logging(self):
        self.permanences = [np.copy(self.permanence)]

    def enable_weights_logging(self):
        self.weights = [np.copy(self.projection.get("weight", format="array").flatten())]

    def __call__(self, runtime: float):
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

            permanence, x, z, mature = self.rule(permanence=self.permanence[c], threshold=self.threshold[c],
                                                 runtime=runtime, x=self.x[j], z=self.z[i],
                                                 permanence_min=self.permanence_min[c],
                                                 neuron_spikes_pre=neuron_spikes_pre,
                                                 neuron_spikes_post_dendrite=neuron_spikes_post_dendrite,
                                                 neuron_spikes_post_soma=neuron_spikes_post_soma)
            self.permanence[c] = permanence
            self.x[j] = x
            self.z[i] = z

            # Todo: Why do we have the if clause regarding the current weight in here?
            if weight[j, i] != self.mature_weight and mature:
                weight[j, i] = self.mature_weight
            elif not mature:
                weight[j, i] = 0
            # Todo: Check if we need to add else for setting weight to 0 for unlearning a connection

        self.projection.set(weight=weight)

        if self.permanences is not None:
            self.permanences.append(np.copy(np.round(self.permanence, 6)))
        if self.weights is not None:
            self.weights.append(np.copy(np.round(self.projection.get("weight", format="array").flatten(), 6)))


class PlasticitySingleNeuron:
    def __init__(self, projection: pynn.Projection, post_somas: pynn.PopulationView):
        self.projection = projection
        print("inside constructor")

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
        print(self.delta_t_min, self.delta_t_max)

        self.x = np.zeros((len(self.projection.pre)))
        self.z = np.zeros((len(self.projection.post)))

    def rule(self, permanence, threshold, x, z, runtime, neuron_spikes_pre, neuron_spikes_post_dendrite,
             neuron_spikes_post_soma):
        mature = False
        for t in np.linspace(0., runtime, int(runtime / self.dt)):
            if self.debug:
                print(t, round(permanence, 5), round(x, 2), round(z, 2))

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

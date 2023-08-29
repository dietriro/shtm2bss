import numpy as np
import quantities
import copy

from matplotlib import pyplot as plt
from tabulate import tabulate
from abc import ABC, abstractmethod

from pynn_brainscales import brainscales2 as pynn
from pynn_brainscales.brainscales2.standardmodels.cells import SpikeSourceArray
from pynn_brainscales.brainscales2.standardmodels.synapses import StaticSynapse

from shtmbss2.plot import plot_membrane


class SHTMBase(ABC):
    def __init__(self, alphabet_size=14, num_neurons_per_symbol=6):
        # Initialize parameters
        self.alphabet_size = alphabet_size
        self.num_neurons_per_symbol = num_neurons_per_symbol

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
        self.init_neurons()
        self.init_connections()
        self.init_external_input()
        self.init_rec_exc()

    def init_neurons(self):
        self.neurons_exc = []
        for i in range(self.alphabet_size):
            dendrites, somas = self.init_neurons_exc()
            self.neurons_exc.append((somas, dendrites))

        self.neurons_inh = self.init_neurons_inh()

        self.neurons_ext = pynn.Population(self.alphabet_size, SpikeSourceArray())

    def init_neurons_exc(self, num_neurons=None):
        if num_neurons is None:
            num_neurons = self.num_neurons_per_symbol

        all_neurons = pynn.Population(num_neurons * 2, pynn.cells.HXNeuron())

        dendrites = pynn.PopulationView(all_neurons, list(range(0, num_neurons * 2, 2)))
        somas = pynn.PopulationView(all_neurons, list(range(1, num_neurons * 2, 2)))

        dendrites.set(
            multicompartment_enable_conductance=True,
            multicompartment_i_bias_nmda=30,
            multicompartment_connect_soma_right=True,
            reset_v_reset=800,
            refractory_period_reset_holdoff=0,
            refractory_period_refractory_time=75,
        )

        somas.set(
            multicompartment_connect_soma=True,
            refractory_period_refractory_time=10,
        )
        somas.record(["spikes"])

        return dendrites, somas

    def init_neurons_inh(self, num_neurons=None):
        if num_neurons is None:
            num_neurons = self.alphabet_size

        pop = pynn.Population(num_neurons, pynn.cells.HXNeuron())

        pop.set(
            refractory_period_refractory_time=2,
        )
        pop.record(["spikes"])

        return pop

    @abstractmethod
    def init_external_input(self):
        pass

    def init_connections(self):
        self.ext_to_exc = []
        for i in range(self.alphabet_size):
            self.ext_to_exc.append(pynn.Projection(
                pynn.PopulationView(self.neurons_ext, [i]),
                self.neurons_exc[i][0],
                pynn.AllToAllConnector(),
                synapse_type=StaticSynapse(weight=200),
                receptor_type="excitatory"))

        self.exc_to_exc = []
        for i in range(self.alphabet_size):
            for j in range(self.alphabet_size):
                if i == j:
                    continue
                self.exc_to_exc.append(pynn.Projection(
                    self.neurons_exc[i][0],
                    self.neurons_exc[j][1],
                    pynn.FixedProbabilityConnector(0.8),
                    synapse_type=StaticSynapse(weight=63),
                    receptor_type="excitatory"))

        self.exc_to_inh = []
        for i in range(self.alphabet_size):
            self.exc_to_inh.append(pynn.Projection(
                self.neurons_exc[i][0],
                pynn.PopulationView(self.neurons_inh, [i]),
                pynn.AllToAllConnector(),
                synapse_type=StaticSynapse(weight=120),
                receptor_type="excitatory"))

        self.inh_to_exc = []
        for i in range(self.alphabet_size):
            self.inh_to_exc.append(pynn.Projection(
                pynn.PopulationView(self.neurons_inh, [i]),
                self.neurons_exc[i][1],
                pynn.AllToAllConnector(),
                synapse_type=StaticSynapse(weight=-63),
                receptor_type="inhibitory"))

    def init_rec_exc(self, alphabet_id=1, neuron_id=1, neuron_type=1):
        # ToDo: What exactly are we recording here? External or excitatory?
        self.rec_neurons_exc = pynn.PopulationView(self.neurons_exc[alphabet_id][neuron_type], [neuron_id])
        self.rec_neurons_exc.record(["v", "spikes"])

    def reset_rec_exc(self):
        self.rec_neurons_exc.record(None)

    def plot_events(self):
        fig, axs = plt.subplots(self.alphabet_size, 1, sharex="all")

        for i in range(self.alphabet_size):
            soma, _ = self.neurons_exc[i]
            local_inhibitory = pynn.PopulationView(self.neurons_inh, [i])

            soma_spikes = [s.base for s in soma.get_data("spikes").segments[-1].spiketrains]
            inhibitory_spikes = [s.base for s in local_inhibitory.get_data("spikes").segments[-1].spiketrains]
            axs[i].eventplot(soma_spikes, label="excitatory", color="C0")
            axs[i].eventplot(inhibitory_spikes, label="inhibitory", color="C1")
            axs[i].set_xlim(0., 0.1)

        axs[-1].set_xlabel("time [ms]")
        fig.text(0.025, 0.5, "symbol", va="center", rotation="vertical")

    def plot_v_exc(self, alphabet_range):
        self.reset_rec_exc()

        for alphabet_id in alphabet_range:
            for neuron_id in range(self.num_neurons_per_symbol):
                self.init_rec_exc(alphabet_id=alphabet_id, neuron_id=neuron_id)
                pynn.run(0.1)
                plot_membrane(self.rec_neurons_exc)
                self.reset_rec_exc()


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
        pop_ref_neuron = pynn.Population(1, pynn.cells.HXNeuron(threshold_v_threshold=300,
                                                                refractory_period_refractory_time=10))
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

    def init_external_input(self):
        spike_times = [[] for i in range(self.alphabet_size)]
        spike_times[0].append(0.04)

        print(spike_times)

        self.neurons_ext.set(spike_times=spike_times)
        print(self.neurons_ext.get("spike_times"))


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

        self.plasticity = Plasticity(self.proj_dendrite_in, post_somas=soma)
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
                print(f'Running emulation step {t+1}/200 for neuron {idx+1}/2')

                pynn.run(self.runtime)

                self.plasticity(self.runtime)
                self.vs[idx].append(pop.get_data("v").segments[-1].irregularlysampledsignals[0])
                self.permanences.append(copy.copy(self.plasticity.permanence))
                self.weight.append(self.proj_dendrite_in.get("weight", format="array"))

                # Todo: Remove if no longer needed
                # spikes[idx].append(pop.get_data("spikes").segments[-1].spiketrains)
                # x.append(self.plasticity.x[0])
                # z.append(self.plasticity.z[0])

            pop.record(None)
            pop.record(["spikes"])

    def test_plasticity(self):
        # Todo: still needed? Which plasticity initialization did we use?
        dendrite, soma, ref_neuron = self.neurons_exc[0]

        test_projection = pynn.Projection(pynn.PopulationView(self.pop_dendrite_in, [0]), dendrite,
                                          pynn.AllToAllConnector(),
                                          pynn.standardmodels.synapses.StaticSynapse(weight=0),
                                          receptor_type="excitatory")

        plasticity = Plasticity(test_projection, post_somas=soma)

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

    def plot_events(self):
        fig, (ax_pre, ax_d, ax_s, ax_p, ax_w) = plt.subplots(5, 1, sharex=True)

        times_dendrite = np.concatenate([np.array(self.vs[0][i].times) + self.runtime * i for i in range(200)])
        vs_dendrite = np.concatenate([np.array(self.vs[0][i]) for i in range(200)])
        times_soma = np.concatenate([np.array(self.vs[1][i].times) + self.runtime * i for i in range(200)])
        vs_soma = np.concatenate([np.array(self.vs[1][i]) for i in range(200)])

        ax_pre.eventplot(
            np.concatenate([np.array(self.pop_dendrite_in.get("spike_times").value) + i * self.runtime for i in range(200)]),
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
        

class Plasticity:
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
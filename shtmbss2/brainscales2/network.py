import numpy as np
import quantities
import copy

from matplotlib import pyplot as plt
from tabulate import tabulate
from abc import ABC

from shtmbss2.brainscales2.config import *
from shtmbss2.core.logging import log
import shtmbss2.common.network as network
from shtmbss2.common.network import NeuronType, RecTypes, symbol_from_label

from pynn_brainscales import brainscales2 as pynn


RECORDING_VALUES = {
    NeuronType.Soma: {RecTypes.SPIKES: "spikes", RecTypes.V: "v"},
    NeuronType.Dendrite: {RecTypes.SPIKES: "spikes", RecTypes.V: "v"},
    NeuronType.Inhibitory: {RecTypes.SPIKES: "spikes", RecTypes.V: "v"}
}


class SHTMBase(network.SHTMBase, ABC):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def init_neurons(self):
        super().init_neurons()

        log.info("Starting preprocessing/calibration...")
        pynn.preprocess()

        for dendrites, somas in self.neurons_exc:
            self.init_neurons_exc_post_preprocess(dendrites, somas)

        if self.p.Experiment.run_add_calib:
            self.run_add_calibration()

    def init_all_neurons_exc(self, num_neurons=None):
        neurons_exc = list()
        for i in range(self.p.Network.num_symbols):
            dendrites, somas = self.init_neurons_exc(num_neurons=num_neurons)
            somas.record([RECORDING_VALUES[NeuronType.Soma][RecTypes.SPIKES]])
            dendrites.record([RECORDING_VALUES[NeuronType.Dendrite][RecTypes.SPIKES]])
            neurons_exc.append((dendrites, somas))

        return neurons_exc

    def init_neurons_exc(self, num_neurons=None):
        if num_neurons is None:
            num_neurons = self.p.Network.num_neurons

        # TODO: remove once pynn_brainscales supports float values directly (bug currently)
        # pynn.cells.CalibHXNeuronCuba.default_parameters.update({"tau_syn_I": 2.})

        all_neurons = pynn.Population(num_neurons * 2, pynn.cells.CalibHXNeuronCuba(
            tau_m=self.p.Neurons.Excitatory.tau_m,
            tau_syn_I=self.p.Neurons.Excitatory.tau_syn_I * num_neurons,
            tau_syn_E=self.p.Neurons.Excitatory.tau_syn_E * num_neurons,
            v_rest=self.p.Neurons.Excitatory.v_rest * num_neurons,
            v_reset=self.p.Neurons.Excitatory.v_reset * num_neurons,
            v_thresh=self.p.Neurons.Excitatory.v_thresh * num_neurons,
            tau_refrac=self.p.Neurons.Excitatory.tau_refrac * num_neurons,
        ))

        dendrites = pynn.PopulationView(all_neurons, slice(NeuronType.Dendrite.ID, num_neurons * 2, 2))
        somas = pynn.PopulationView(all_neurons, slice(NeuronType.Soma.ID, num_neurons * 2, 2))


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

        pop.record(RECORDING_VALUES[NeuronType.Inhibitory][RecTypes.SPIKES])

        return pop

    def run_add_calibration(self, v_rest_calib=275):
        self.reset_rec_exc()
        for alphabet_id in range(4):
            for neuron_id in range(15):
                log.debug(alphabet_id, neuron_id, "start")
                for v_leak in range(450, 600, 10):
                    self.neurons_exc[alphabet_id][1].actual_hwparams[neuron_id].leak.v_leak = v_leak
                    self.init_rec_exc(alphabet_id=alphabet_id, neuron_id=neuron_id, neuron_type=NeuronType.Soma)
                    pynn.run(0.02)
                    membrane = self.rec_neurons_exc.get_data("v").segments[-1].irregularlysampledsignals[0].base[1]
                    self.reset_rec_exc()
                    if v_rest_calib <= membrane.mean():
                        for v_leak_inner in range(v_leak - 10, v_leak + 10, 1):
                            self.neurons_exc[alphabet_id][1].actual_hwparams[neuron_id].leak.v_leak = v_leak_inner
                            self.init_rec_exc(alphabet_id=alphabet_id, neuron_id=neuron_id, neuron_type=NeuronType.Soma)
                            pynn.run(0.02)
                            membrane = \
                                self.rec_neurons_exc.get_data("v").segments[-1].irregularlysampledsignals[0].base[1]
                            self.reset_rec_exc()
                            if v_rest_calib <= membrane.mean():
                                log.debug(alphabet_id, neuron_id, v_leak_inner, membrane.mean())
                                break
                        break

    def init_rec_exc(self, alphabet_id=1, neuron_id=1, neuron_type=NeuronType.Soma):
        self.rec_neurons_exc = pynn.PopulationView(self.neurons_exc[alphabet_id][neuron_type.ID], [neuron_id])
        self.rec_neurons_exc.record([RECORDING_VALUES[neuron_type][RecTypes.V],
                                     RECORDING_VALUES[neuron_type][RecTypes.SPIKES]])

    def reset_rec_exc(self):
        self.rec_neurons_exc.record(None)

    def get_neurons(self, neuron_type, symbol_id=None):
        if symbol_id is None:
            log.error("Cannot get neurons with symbol_id None")
            return

        if neuron_type == NeuronType.Inhibitory:
            return pynn.PopulationView(self.neurons_inh, [symbol_id])
        elif neuron_type in [NeuronType.Dendrite, NeuronType.Soma]:
            return self.neurons_exc[symbol_id][neuron_type.ID]

    def get_neuron_data(self, neuron_type, neurons=None, value_type="spikes", symbol_id=None, neuron_id=None,
                        runtime=None, dtype=None):
        """
        Returns the recorded data for the given neuron type or neuron population.

        :param neuron_type: The type of the neuron.
        :type neuron_type: Union[NeuronType.Dendrite, NeuronType.Soma, NeuronType.Inhibitory]
        :param neurons: Optionally, the neuron population for which the data should be returned
        :type neurons: pynn.Population
        :param value_type: The value type (RecType) of the data to be returned ['spikes', 'v'].
        :type value_type: str
        :param symbol_id: The id of the symbol for which the data should be returned.
        :type symbol_id: int
        :param neuron_id: The index of the neuron within its population.
        :type neuron_id: int
        :param runtime: The runtime used during the last experiment.
        :type runtime: float
        :param dtype: The structure type of the returned data. Default is None, i.e. a SpikeTrainList.
        :type dtype: Union[list, np.ndarray, None]
        :return: The specified data, recorded from the neuron for the past experiment.
        :rtype:
        """
        if neurons is None:
            neurons = self.get_neurons(neuron_type, symbol_id=symbol_id)

        if value_type == RecTypes.SPIKES:
            data = neurons.get_data(RECORDING_VALUES[neuron_type][value_type]).segments[-1].spiketrains

            if dtype is np.ndarray:
                data = np.array(data.multiplexed).transpose()
                # ToDo: Doub-check if the following line needs to be added here as well to convert from id to index
                # data[:, 0] = neurons.id_to_index(data[:, 0])
            elif dtype is list:
                data = [s.base for s in data]
        elif value_type == RecTypes.V:
            if runtime is None:
                log.error(f"Could not retrieve voltage data for {neuron_type}")
                return None

            self.reset_rec_exc()
            self.init_rec_exc(alphabet_id=symbol_id, neuron_id=neuron_id, neuron_type=neuron_type)

            pynn.run(runtime)

            data = self.rec_neurons_exc.get_data("v").segments[-1].irregularlysampledsignals[0]
        else:
            log.error(f"Error retrieving neuron data! Unknown value_type: '{value_type}'.")
            return None
        return data


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
        self.init_rec_exc(alphabet_id=0, neuron_id=0, neuron_type=NeuronType.Soma)

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

    def init_external_input(self, init_recorder=False):
        # Define input spikes, population and connections for dendritic coincidence spikes
        spikes_coincidence = self.wait_before_experiment + np.arange(10) * self.isi
        spikes_coincidence = np.array(spikes_coincidence.rescale(quantities.ms))
        self.pop_dendrite_in = pynn.Population(2,  # ToDo: Why 2 neurons?
                                               pynn.cells.SpikeSourceArray(spike_times=spikes_coincidence))

        # Define input spikes, population and connections for somatic spikes
        self.pop_soma_in = pynn.Population(4,  # ToDo: Why 4 neurons?
                                           pynn.cells.SpikeSourceArray(spike_times=[0.025]))

    def init_connections(self):
        dendrite, soma, ref_neuron = self.neurons_exc[0]

        self.proj_dendrite_in = pynn.Projection(self.pop_dendrite_in, dendrite,
                                                pynn.AllToAllConnector(),
                                                synapse_type=pynn.synapses.StaticSynapse(weight=63),
                                                receptor_type="excitatory")

        self.proj_soma_in = pynn.Projection(self.pop_soma_in, soma,
                                            pynn.AllToAllConnector(),
                                            synapse_type=pynn.synapses.StaticSynapse(weight=40),
                                            receptor_type="excitatory")

        self.proj_ref_in = pynn.Projection(self.pop_soma_in, ref_neuron,
                                           pynn.AllToAllConnector(),
                                           synapse_type=pynn.synapses.StaticSynapse(weight=40),
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


class SHTMPlasticity(SHTMSingleNeuron):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.weight = None
        self.permanences = None
        self.vs = None
        self.plasticity = None

    def init_external_input(self, init_recorder=False):
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

        if runtime is None:
            runtime = self.p.Experiment.runtime
        else:
            self.p.Experiment.runtime = runtime

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

                pynn.run(runtime)

                self.plasticity(runtime)

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

        times_dendrite = np.concatenate(
            [np.array(self.vs[0][i].times) + self.p.Experiment.runtime * i for i in range(200)])
        vs_dendrite = np.concatenate([np.array(self.vs[0][i]) for i in range(200)])
        times_soma = np.concatenate([np.array(self.vs[1][i].times) + self.p.Experiment.runtime * i for i in range(200)])
        vs_soma = np.concatenate([np.array(self.vs[1][i]) for i in range(200)])

        ax_pre.eventplot(
            np.concatenate(
                [np.array(self.pop_dendrite_in.get("spike_times").value) + i * self.p.Experiment.runtime for i in
                 range(200)]),
            label="pre", lw=.2)
        ax_pre.legend()
        ax_d.plot(times_dendrite, vs_dendrite, label="dendrite", lw=.2)
        ax_d.legend()
        ax_s.plot(times_soma, vs_soma, label="soma", lw=.2)
        ax_s.legend()
        ax_p.plot(np.linspace(0., self.p.Experiment.runtime * 200., 200), self.permanences[:200], label="permanence")
        ax_p.set_ylabel("P [a.u.]")
        # ax_p.legend()
        ax_w.plot(np.linspace(0., self.p.Experiment.runtime * 200., 200),
                  np.array([w for w in np.array(self.weight[:200]).squeeze().T if any(w)]).T, label="weight")
        ax_w.set_ylabel("w [a.u.]")
        # ax_w.legend()
        ax_w.set_xlabel("time [ms]")
        plt.savefig("../evaluation/plasticity.pdf")


class SHTMTotal(SHTMBase, network.SHTMTotal):
    def __init__(self, **kwargs):
        super().__init__(plasticity_cls=Plasticity, **kwargs)


class Plasticity(network.Plasticity):
    def __init__(self, projection: pynn.Projection, post_somas, shtm, **kwargs):
        super().__init__(projection, post_somas, shtm, **kwargs)

    def get_connection_id_pre(self, connection):
        return connection.presynaptic_index

    def get_connection_id_post(self, connection):
        return connection.postsynaptic_index

    def get_connections(self):
        return self.projection.connections


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

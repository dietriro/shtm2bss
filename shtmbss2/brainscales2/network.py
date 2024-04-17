import numpy as np
import quantities
import copy

from matplotlib import pyplot as plt
from tabulate import tabulate
from abc import ABC

from shtmbss2.brainscales2.config import *
from shtmbss2.brainscales2.patches import patch_pynn_calibration
from shtmbss2.brainscales2.plasticity import PlasticityOnChip
from shtmbss2.core.logging import log
from shtmbss2.core.helpers import id_to_symbol
import shtmbss2.common.network as network
from shtmbss2.common.config import NeuronType, RecTypes

from pynn_brainscales import brainscales2 as pynn
from pynn_brainscales.brainscales2 import simulator, Projection, PopulationView
from pynn_brainscales.brainscales2.connectors import AllToAllConnector, FixedNumberPreConnector
from pynn_brainscales.brainscales2.standardmodels.synapses import StaticSynapse
from pyNN.random import NumpyRNG
from dlens_vx_v3 import sta, halco, hal, lola

RECORDING_VALUES = {
    NeuronType.Soma: {RecTypes.SPIKES: "spikes", RecTypes.V: "v"},
    NeuronType.Dendrite: {RecTypes.SPIKES: "spikes", RecTypes.V: "v"},
    NeuronType.Inhibitory: {RecTypes.SPIKES: "spikes", RecTypes.V: "v"}
}


class SHTMBase(network.SHTMBase, ABC):

    def __init__(self, experiment_type=ExperimentType.EVAL_SINGLE, experiment_subnum=None, instance_id=None,
                 seed_offset=None, p=None, use_on_chip_plasticity=False, **kwargs):
        super().__init__(experiment_type=experiment_type, experiment_subnum=experiment_subnum, instance_id=instance_id,
                         seed_offset=seed_offset, p=p, **kwargs)
        self.use_on_chip_plasticity = use_on_chip_plasticity
        self.exc_to_exc_soma_to_soma_dummy = None
        self.exc_to_exc_dendrite_to_soma_dummy = None
        self.plasticity_rule = None

    def load_params(self, **kwargs):
        super().load_params(**kwargs)

        # ToDo: Fix this and remove temporary hack
        self.p.Neurons.Dendrite.theta_dAP = self.p.Neurons.Excitatory.v_thresh[NeuronType.Dendrite.ID]
        self.p.Neurons.Dendrite.I_p = self.p.Neurons.Excitatory.v_reset[NeuronType.Dendrite.ID]
        self.p.Neurons.Dendrite.tau_dAP = self.p.Neurons.Excitatory.tau_refrac[NeuronType.Dendrite.ID]

        patch_pynn_calibration(
            self.p.Calibration.padi_bus_dacen_extension,
            self.p.Calibration.correlation_amplitude,
            self.p.Calibration.correlation_time_constant,
        )

    def init_plasticity_rule(self, start_time: float = None, period: float = None, runtime: float = 0.,
                             permanence_threshold: int = 100, mature_weight: int = 63):
        if not (0 <= mature_weight <= 63):
            raise ValueError("Mature weight needs to be in [0, 63].")

        if start_time is None:
            start_time = self.p.Plasticity.execution_start
        else:
            self.p.Plasticity.execution_start = start_time

        if period is None:
            period = self.p.Plasticity.execution_interval
        else:
            self.p.Plasticity.execution_interval = period

        timer = pynn.plasticity_rules.Timer(start=start_time, period=period,
                                            num_periods=int((runtime - start_time) / period) + 1)

        self.plasticity_rule = PlasticityOnChip(
            timer=timer,
            num_neurons=self.p.Network.num_symbols * self.p.Network.num_neurons,
            permanence_threshold=int(self.p.Plasticity.permanence_threshold),
            w_mature=int(self.p.Plasticity.w_mature),
            target_rate_h=self.p.Plasticity.target_rate_h,
            lambda_plus=self.p.Plasticity.lambda_plus,
            lambda_minus=self.p.Plasticity.lambda_minus,
            lambda_h=self.p.Plasticity.lambda_h,
            learning_factor=self.p.Plasticity.learning_factor,
            p_exc_exc=self.p.Synapses.p_exc_exc,
            delta_t_max=self.p.Plasticity.delta_t_max,
            tau_plus=self.p.Plasticity.tau_plus,
            correlation_threshold=self.p.Plasticity.correlation_threshold
        )

        # read-out permanence to retain between epochs
        pynn.simulator.state.injected_readout.ppu_symbols = {
            "permanences",
            "dummy"
        }

        # reset neuron counters
        pynn.simulator.state.injection_pre_realtime = sta.PlaybackProgramBuilder()
        for coord in halco.iter_all(halco.SpikeCounterResetOnDLS):
            pynn.simulator.state.injection_pre_realtime.write(coord, hal.SpikeCounterReset())
        pynn.simulator.state.injection_pre_realtime.block_until(halco.BarrierOnFPGA(), hal.Barrier.omnibus)

    def init_neurons(self):
        super().init_neurons()

        log.info("Starting preprocessing/calibration...")
        pynn.preprocess()

        for dendrites, somas in self.neurons_exc:
            self.init_neurons_exc_post_preprocess(dendrites, somas)

        if self.p.Experiment.run_add_calib:
            self.run_add_calibration()

    def init_all_neurons_exc(self, num_neurons=None):
        all_dendrites, all_somas = self.init_neurons_exc(self.p.Network.num_symbols * self.p.Network.num_neurons)
        self.neurons_exc_all = (all_dendrites, all_somas)

        neurons_exc = list()
        for i in range(self.p.Network.num_symbols):
            dendrites = pynn.PopulationView(all_dendrites,
                                            slice(i * self.p.Network.num_neurons, (i + 1) * self.p.Network.num_neurons))
            somas = pynn.PopulationView(all_somas,
                                        slice(i * self.p.Network.num_neurons, (i + 1) * self.p.Network.num_neurons))
            somas.record([RECORDING_VALUES[NeuronType.Soma][RecTypes.SPIKES]])
            dendrites.record([RECORDING_VALUES[NeuronType.Dendrite][RecTypes.SPIKES]])
            neurons_exc.append((dendrites, somas))

        return neurons_exc

    def init_neurons_exc(self, num_neurons=None):
        if num_neurons is None:
            num_neurons = self.p.Network.num_neurons

        # TODO: remove once pynn_brainscales supports float values directly (bug currently)
        # pynn.cells.CalibHXNeuronCuba.default_parameters.update({"tau_syn_I": 2.})

        dendrites = pynn.Population(num_neurons, pynn.cells.CalibHXNeuronCuba(
            plasticity_rule=self.plasticity_rule,
            tau_m=self.p.Neurons.Excitatory.tau_m[0],
            tau_syn_I=self.p.Neurons.Excitatory.tau_syn_I[0],
            tau_syn_E=self.p.Neurons.Excitatory.tau_syn_E[0],
            v_rest=self.p.Neurons.Excitatory.v_rest[0],
            v_reset=self.p.Neurons.Excitatory.v_reset[0],
            v_thresh=self.p.Neurons.Excitatory.v_thresh[0],
            tau_refrac=self.p.Neurons.Excitatory.tau_refrac[0],
        ))

        somas = pynn.Population(num_neurons, pynn.cells.CalibHXNeuronCuba(
            plasticity_rule=self.plasticity_rule,
            tau_m=self.p.Neurons.Excitatory.tau_m[1],
            tau_syn_I=self.p.Neurons.Excitatory.tau_syn_I[1],
            tau_syn_E=self.p.Neurons.Excitatory.tau_syn_E[1],
            v_rest=self.p.Neurons.Excitatory.v_rest[1],
            v_reset=self.p.Neurons.Excitatory.v_reset[1],
            v_thresh=self.p.Neurons.Excitatory.v_thresh[1],
            tau_refrac=self.p.Neurons.Excitatory.tau_refrac[1],
        ))

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
                    self.run_sim(0.02)
                    membrane = self.rec_neurons_exc.get_data("v").segments[-1].irregularlysampledsignals[0].base[1]
                    self.reset_rec_exc()
                    if v_rest_calib <= membrane.mean():
                        for v_leak_inner in range(v_leak - 10, v_leak + 10, 1):
                            self.neurons_exc[alphabet_id][1].actual_hwparams[neuron_id].leak.v_leak = v_leak_inner
                            self.init_rec_exc(alphabet_id=alphabet_id, neuron_id=neuron_id, neuron_type=NeuronType.Soma)
                            self.run_sim(0.02)
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

    def reset(self, store_to_cache=False):
        pynn.reset()

        # disable caching of segments (takes lots of time and space)
        if not store_to_cache:
            for recorder in simulator.state.recorders:
                recorder.clear_flag = True

        # # Restart recording of spikes
        # for i_symbol in range(self.p.Network.num_symbols):
        #     somas = pynn.PopulationView(self.neurons_exc[i_symbol], slice(NeuronType.Soma.ID,
        #                                                                   self.p.Network.num_neurons * 2, 2))
        #     dendrites = pynn.PopulationView(self.neurons_exc[i_symbol], slice(NeuronType.Dendrite.ID,
        #                                                                       self.p.Network.num_neurons * 2, 2))
        #     somas.record([RECORDING_VALUES[NeuronType.Soma][RecTypes.SPIKES]])
        #     dendrites.record([RECORDING_VALUES[NeuronType.Dendrite][RecTypes.SPIKES]])

        self.run_state = False

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
                spike_times = data.multiplexed
                if len(spike_times[0]) > 0:
                    data = np.array(spike_times).transpose()
                else:
                    data = np.empty((0, 2))
            elif dtype is list:
                data = [s.base for s in data]
        elif value_type == RecTypes.V:
            if runtime is None:
                log.error(f"Could not retrieve voltage data for {neuron_type}")
                return None

            self.reset()

            self.reset_rec_exc()
            self.init_rec_exc(alphabet_id=symbol_id, neuron_id=neuron_id, neuron_type=neuron_type)

            self.run_sim(runtime)

            data = self.rec_neurons_exc.get_data("v").segments[-1].irregularlysampledsignals[0]
        else:
            log.error(f"Error retrieving neuron data! Unknown value_type: '{value_type}'.")
            return None
        return data


class SHTMSingleNeuron(SHTMBase):
    def __init__(self, instance_id=None, seed_offset=None, **kwargs):
        super().__init__(instance_id=instance_id, seed_offset=seed_offset, **kwargs)

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
    def __init__(self, experiment_type=ExperimentType.EVAL_SINGLE, experiment_subnum=None, instance_id=None,
                 seed_offset=None, p=None,
                 **kwargs):
        super().__init__(experiment_type=experiment_type, experiment_subnum=experiment_subnum,
                         plasticity_cls=Plasticity, instance_id=instance_id,
                         seed_offset=seed_offset, p=p, **kwargs)

    def init_connections(self, exc_to_exc=None, exc_to_inh=None, debug=None):
        if not self.use_on_chip_plasticity:
            super().init_connections(exc_to_exc=exc_to_exc, exc_to_inh=exc_to_inh, debug=debug)
        else:
            self.ext_to_exc = []
            for i in range(self.p.Network.num_symbols):
                self.ext_to_exc.append(Projection(
                    PopulationView(self.neurons_ext, [i]),
                    self.get_neurons(NeuronType.Soma, symbol_id=i),
                    AllToAllConnector(),
                    synapse_type=StaticSynapse(weight=self.p.Synapses.w_ext_exc, delay=self.p.Synapses.delay_ext_exc),
                    receptor_type=self.p.Synapses.receptor_ext_exc))

            self.exc_to_exc = []
            self.exc_to_exc_soma_to_soma_dummy = []
            self.exc_to_exc_dendrite_to_soma_dummy = []
            num_connections = int(self.p.Network.num_neurons * self.p.Synapses.p_exc_exc)
            weight = self.p.Synapses.w_exc_exc if exc_to_exc is None else exc_to_exc
            seed = self.p.Experiment.seed_offset
            if self.instance_id is not None:
                seed += self.instance_id * self.p.Network.num_symbols ** 2
            all_dendrites, all_somas = self.neurons_exc_all
            self.exc_to_exc.append(Projection(
                all_somas,
                all_dendrites,
                AllToAllConnector(),
                synapse_type=pynn.standardmodels.synapses.PlasticSynapse(weight=weight,
                                                                         delay=self.p.Synapses.delay_exc_exc,
                                                                         plasticity_rule=self.plasticity_rule),
                receptor_type=self.p.Synapses.receptor_exc_exc,
                label=f"exc-exc_soma_to_dendrite"))
            self.exc_to_exc_soma_to_soma_dummy.append(Projection(
                all_somas,
                all_somas,
                AllToAllConnector(),
                synapse_type=pynn.standardmodels.synapses.PlasticSynapse(weight=0,
                                                                         plasticity_rule=self.plasticity_rule),
                receptor_type=self.p.Synapses.receptor_exc_exc,
                label=f"exc-exc_soma_to_soma-dummy"))
            self.exc_to_exc_dendrite_to_soma_dummy.append(Projection(
                all_dendrites,
                all_somas,
                AllToAllConnector(),
                synapse_type=pynn.standardmodels.synapses.PlasticSynapse(weight=0,
                                                                         plasticity_rule=self.plasticity_rule),
                receptor_type=self.p.Synapses.receptor_exc_exc,
                label=f"exc-exc_dendrite_to_soma-dummy"))

            self.exc_to_inh = []
            for i in range(self.p.Network.num_symbols):
                weight = self.p.Synapses.w_exc_inh if exc_to_inh is None else exc_to_inh[i]
                self.exc_to_inh.append(Projection(
                    self.get_neurons(NeuronType.Soma, symbol_id=i),
                    PopulationView(self.neurons_inh, [i]),
                    AllToAllConnector(),
                    synapse_type=StaticSynapse(weight=weight, delay=self.p.Synapses.delay_exc_inh),
                    receptor_type=self.p.Synapses.receptor_exc_inh))

            self.inh_to_exc = []
            for i in range(self.p.Network.num_symbols):
                self.inh_to_exc.append(Projection(
                    PopulationView(self.neurons_inh, [i]),
                    self.get_neurons(NeuronType.Soma, symbol_id=i),
                    AllToAllConnector(),
                    synapse_type=StaticSynapse(weight=self.p.Synapses.w_inh_exc, delay=self.p.Synapses.delay_inh_exc),
                    receptor_type=self.p.Synapses.receptor_inh_exc))

            self.con_plastic = list()
            for i in range(self.p.Network.num_symbols * (self.p.Network.num_symbols - 1)):
                self.con_plastic.append(OnChipPlasticityDummy(self.exc_to_exc[0], self, id=i))

    def __update_plasticity(self):
        num_neurons_total = self.p.Network.num_neurons * self.p.Network.num_symbols
        num_neurons = self.p.Network.num_neurons

        # calculate weights
        weights = self.exc_to_exc[0].get("weight", format="array")

        # calculate rates
        rates = np.array(self.exc_to_exc_soma_to_soma_dummy[0].get_data("data")[-1].data)
        rates = rates.reshape((num_neurons_total, num_neurons_total))

        # calculate permanences
        permanences = np.array(self.exc_to_exc_dendrite_to_soma_dummy[0].get_data("data")[-1].data).flatten()
        permanences = permanences.reshape((num_neurons_total, num_neurons_total))

        # calculate correlations
        x = np.array(self.exc_to_exc_soma_to_soma_dummy[0].get_data("correlation")[-1].data)
        x = x.reshape((num_neurons_total, num_neurons_total))
        z = np.array(self.exc_to_exc_dendrite_to_soma_dummy[0].get_data("correlation")[-1].data)
        z = z.reshape((num_neurons_total, num_neurons_total))

        # calculate values for each connection pair (symbol-i to symbol-k)
        i_plastic = 0
        for i in range(self.p.Network.num_symbols):
            for k in range(self.p.Network.num_symbols):
                if i == k:
                    continue

                i_neuron = i * num_neurons
                k_neuron = k * num_neurons

                weights_tmp = weights[i_neuron:i_neuron + num_neurons, k_neuron:k_neuron + num_neurons].flatten().copy()
                rates_tmp = rates[i_neuron:i_neuron + num_neurons, k_neuron:k_neuron + num_neurons].flatten().copy()
                permanences_tmp = permanences[i_neuron:i_neuron + num_neurons,
                                              k_neuron:k_neuron + num_neurons].flatten().copy()
                x_tmp = x[i_neuron:i_neuron + num_neurons, k_neuron:k_neuron + num_neurons].flatten().copy()
                z_tmp = z[i_neuron:i_neuron + num_neurons, k_neuron:k_neuron + num_neurons].flatten().copy()

                self.con_plastic[i_plastic].update_values(weights_tmp, rates_tmp, permanences_tmp, x_tmp, z_tmp)

                i_plastic += 1

    def run(self, runtime=None, steps=None, plasticity_enabled=True, store_to_cache=False, store_plasticity_values=True,
            dyn_exc_inh=False, run_type=RunType.SINGLE):
        if not self.use_on_chip_plasticity:
            super().run(runtime=runtime, steps=steps, plasticity_enabled=plasticity_enabled, dyn_exc_inh=dyn_exc_inh,
                        run_type=run_type)
        else:
            if runtime is None:
                runtime = self.p.Experiment.runtime
            if steps is None:
                steps = self.p.Experiment.episodes
            self.p.Experiment.episodes = 0

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

            initial_permanences = lola.ExternalPPUMemoryBlock(
                halco.ExternalPPUMemoryBlockSize(256 * 60))
            # TODO: support drawing initial values between min and max
            initial_permanences.bytes = [hal.ExternalPPUMemoryByte(
                hal.ExternalPPUMemoryByte.Value(self.p.Plasticity.permanence_init_min))] * 256 * 60
            pynn.simulator.state.injected_config.ppu_symbols = {
                "permanences": initial_permanences}

            run_length = self.p.Experiment.runtime / self.p.Encoding.num_repetitions
            performance_t_min = ((np.ceil(self.p.Plasticity.execution_start / run_length) - 2) * run_length
                                 + self.p.Encoding.t_exc_start)

            for t in range(steps):
                log.info(f'Running emulation step {t + 1}/{steps}')

                # reset the simulator and the network state if not first run
                if self.run_state:
                    self.reset(store_to_cache)

                # set start time to 0.0 because
                # - nest is reset and always starts with 0.0
                # - bss2 resets the time itself after each run to 0.0
                sim_start_time = 0.0
                log.detail(f"Current time: {sim_start_time}")

                self.run_sim(runtime=runtime)

                # retrieve permanence data from plasticity processor and inject into next execution
                pynn.simulator.state.injected_config.ppu_symbols = {
                    "permanences": pynn.get_post_realtime_read_ppu_symbols()["permanences"]}

                # update weight data in projection
                assert len(self.exc_to_exc) == 1
                weights_post = np.array(self.exc_to_exc[0].get_data("data")[-1].data).reshape((len(self.exc_to_exc[0])))
                self.exc_to_exc[0].set(weight=weights_post)

                self._retrieve_neuron_data()

                # expose data in plasticity rule dummy (one call suffices)
                if store_plasticity_values:
                    self.__update_plasticity()

                if self.p.Performance.compute_performance:
                    self.performance.compute(neuron_events=self.neuron_events, method=self.p.Performance.method,
                                             t_min=performance_t_min)

                if plasticity_enabled:
                    if run_type == RunType.MULTI:
                        log.warn(
                            f"Multi-core version of plasticity calculation is currently not working. Please choose the "
                            f"single-core version. Not calculating plasticity.")
                        # self.__run_plasticity_parallel(runtime, sim_start_time, dyn_exc_inh=dyn_exc_inh)
                    elif run_type == RunType.SINGLE:
                        self.__run_plasticity_singular(runtime, sim_start_time, dyn_exc_inh=dyn_exc_inh)

                if self.p.Experiment.save_auto and self.p.Experiment.save_auto_epoches > 0:
                    if (t + 1) % self.p.Experiment.save_auto_epoches == 0:
                        self.p.Experiment.episodes = self.experiment_episodes + t + 1
                        self.save_full_state()

            self.experiment_episodes += steps
            self.p.Experiment.episodes = self.experiment_episodes

            if self.p.Experiment.save_final or self.p.Experiment.save_auto:
                self.save_full_state()

    def plot_data_overview(self, data_types="all"):
        # define structure to hold all data
        plot_data = dict()
        # define total number of neurons
        num_neurons_total = self.p.Network.num_neurons * self.p.Network.num_symbols

        # retrieve current data
        plot_data["cor_soma-soma"] = np.array(self.exc_to_exc_soma_to_soma_dummy[0].get_data("correlation")[-1].data)
        plot_data["cor_dend-soma"] = np.array(self.exc_to_exc_dendrite_to_soma_dummy[0].get_data("correlation")[-1].data)
        plot_data["permanences"] = np.array(self.exc_to_exc_dendrite_to_soma_dummy[0].get_data("data")[-1].data)
        plot_data["weights"] = self.exc_to_exc[0].get("weight", format="array")

        # define plotted data types
        if data_types is None or (type(data_types) is str and data_types == "all"):
            data_types = plot_data.keys()
        elif type(data_types) is list():
            pass
        elif type(data_types) is str:
            if data_types in plot_data.keys():
                data_types = [data_types]
            else:
                log.error(f"Value of parameter 'data_types' is not known: {data_types}")
                return
        else:
            log.error(f"Parameter 'data_types' has an unsupported type '{type(data_types)}'.")
            return

        # plot data
        fig, axs = plt.subplots(1, len(data_types), figsize=(len(data_types)*5, 5))
        for i_plot, data_name in enumerate(data_types):
            data_arr = plot_data[data_name].reshape((num_neurons_total, num_neurons_total))

            axs[i_plot].imshow(data_arr, interpolation='nearest')

            # Major ticks
            ticks_major = np.arange(self.p.Network.num_neurons/2, num_neurons_total, self.p.Network.num_neurons)
            axs[i_plot].set_xticks(ticks_major)
            axs[i_plot].set_yticks(ticks_major)

            symbols = [id_to_symbol(sym_i) for sym_i in range(self.p.Network.num_symbols)]
            axs[i_plot].set_xticklabels(symbols)
            axs[i_plot].set_yticklabels(symbols)

            # Minor ticks
            axs[i_plot].set_xticks(np.arange(-0.5, num_neurons_total + 0.5, self.p.Network.num_neurons), minor=True)
            axs[i_plot].set_yticks(np.arange(-0.5, num_neurons_total + 0.5, self.p.Network.num_neurons), minor=True)

            axs[i_plot].grid(which='minor', color='w', linestyle='-', linewidth=1)

            axs[i_plot].set_title(data_name, fontsize=20)
            axs[i_plot].set_xlabel("Target", fontsize=16)
            axs[i_plot].set_ylabel("Source", fontsize=16)

            i_plot += 1

        fig.show()

    def plot_permanence_histogram(self, bin_width=5):
        # Get data
        num_neurons_total = self.p.Network.num_neurons * self.p.Network.num_symbols
        soma_cor = np.array(self.exc_to_exc_soma_to_soma_dummy[0].get_data("correlation")[-1].data)
        soma_cor = soma_cor.reshape((num_neurons_total, num_neurons_total))

        # Plot histogram
        data = soma_cor.flatten()
        plt.hist(data, bins=range(min(data), max(data) + bin_width, bin_width))


class Plasticity(network.Plasticity):
    def __init__(self, projection: pynn.Projection, post_somas, shtm, index, **kwargs):
        super().__init__(projection, post_somas, shtm, index, **kwargs)

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
        self.permanence_threshold = np.ones((len(self.projection))) * 20.
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


class OnChipPlasticityDummy(ABC):
    def __init__(self, projection: Projection, shtm, id=0):
        # custom objects
        self.projection = projection
        self.shtm: SHTMTotal = shtm

        self.permanences = []
        self.rates = []
        self.permanence = None
        self.permanence_min = 0
        self.weights = []
        self.x = []
        self.z = []

        self.id = id

    def update_values(self, weights, rates, permanences, x, z):
        # save weights
        self.weights.append(weights)

        # save rates
        self.rates.append(rates)

        # save permanences
        self.permanences.append(permanences)
        self.permanence = self.permanences[-1]

        # save correlations
        self.x.append(x)
        self.z.append(z)

    def enable_permanence_logging(self):
        pass

    def enable_weights_logging(self):
        pass

    def get_connection_ids(self, connection_id):
        connection_ids = (f"{self.get_connection_id_pre(self.get_connections()[connection_id])}>"
                          f"{self.get_connection_id_post(self.get_connections()[connection_id])}")
        return connection_ids

    def get_connection_id_pre(self, connection):
        return connection.presynaptic_index

    def get_connection_id_post(self, connection):
        return connection.postsynaptic_index

    def get_all_connection_ids(self):
        connection_ids = []
        for con in self.get_connections():
            connection_ids.append(f"{self.get_connection_id_pre(con)}>{self.get_connection_id_post(con)}")
        return connection_ids

    def get_connections(self):
        return self.projection.connections

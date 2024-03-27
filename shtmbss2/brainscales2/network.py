import numpy as np
import quantities
import copy

from matplotlib import pyplot as plt
from tabulate import tabulate
from abc import ABC

from shtmbss2.brainscales2.config import *
from shtmbss2.brainscales2.patches import patch_pynn_calibration
from shtmbss2.core.logging import log
import shtmbss2.common.network as network
from shtmbss2.common.config import NeuronType, RecTypes

from pynn_brainscales import brainscales2 as pynn


RECORDING_VALUES = {
    NeuronType.Soma: {RecTypes.SPIKES: "spikes", RecTypes.V: "v"},
    NeuronType.Dendrite: {RecTypes.SPIKES: "spikes", RecTypes.V: "v"},
    NeuronType.Inhibitory: {RecTypes.SPIKES: "spikes", RecTypes.V: "v"}
}


class SHTMBase(network.SHTMBase, ABC):

    def __init__(self, experiment_type=ExperimentType.EVAL_SINGLE, experiment_subnum=None, instance_id=None, seed_offset=None, p=None,
                 **kwargs):
        super().__init__(experiment_type=experiment_type, experiment_subnum=experiment_subnum, instance_id=instance_id,
                         seed_offset=seed_offset, p=p, **kwargs)

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

    def reset(self):
        pynn.reset()

        # Restart recording of spikes
        for i_symbol in range(self.p.Network.num_symbols):
            somas = pynn.PopulationView(self.neurons_exc[i_symbol], slice(NeuronType.Soma.ID,
                                                                          self.p.Network.num_neuron * 2, 2))
            dendrites = pynn.PopulationView(self.neurons_exc[i_symbol], slice(NeuronType.Dendrite.ID,
                                                                              self.p.Network.num_neuron * 2, 2))
            somas.record([RECORDING_VALUES[NeuronType.Soma][RecTypes.SPIKES]])
            dendrites.record([RECORDING_VALUES[NeuronType.Dendrite][RecTypes.SPIKES]])

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


class SHTMTotal(SHTMBase, network.SHTMTotal):
    def __init__(self, experiment_type=ExperimentType.EVAL_SINGLE, experiment_subnum=None, instance_id=None, seed_offset=None, p=None,
                 **kwargs):
        super().__init__(experiment_type=experiment_type, experiment_subnum=experiment_subnum, plasticity_cls=Plasticity, instance_id=instance_id,
                         seed_offset=seed_offset, p=p, **kwargs)


class Plasticity(network.Plasticity):
    def __init__(self, projection: pynn.Projection, post_somas, shtm, index, **kwargs):
        super().__init__(projection, post_somas, shtm, index, **kwargs)

    def get_connection_id_pre(self, connection):
        return connection.presynaptic_index

    def get_connection_id_post(self, connection):
        return connection.postsynaptic_index

    def get_connections(self):
        return self.projection.connections

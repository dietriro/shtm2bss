from matplotlib import pyplot as plt


def plot_membrane(membrane_recorded, label=None):
    membrane = membrane_recorded.get_data("v").segments[-1].irregularlysampledsignals[0]
    plt.plot(membrane.times, membrane, alpha=0.5, label=label)


def plot_dendritic_events(axis, spikes_dend, spikes_post, color, label, y_offset=1.0, seq_start=0, seq_end=None):
    for i_neuron in range(len(spikes_dend)):
        if len(spikes_dend[i_neuron]) <= 0:
            continue
        spikes_dend_i = np.array(spikes_dend[i_neuron])
        spikes_dend_i = spikes_dend_i[np.where(np.logical_and(spikes_dend_i > seq_start, spikes_dend_i < seq_end))]
        if len(spikes_dend_i) <= 0:
            continue
        spike_dend_i = spikes_dend_i[0]

        if len(spikes_post[i_neuron]) <= 0:
            spike_post_i = spike_dend_i + 0.06
        else:
            spikes_post_i = np.array(spikes_post[i_neuron])
            spikes_post_i = spikes_post_i[
                np.where(np.logical_and(spikes_post_i > seq_start, spikes_post_i < seq_end))]
            if len(spikes_post_i) <= 0:
                spike_post_i = spike_dend_i + 0.06
            else:
                spike_post_i = min(spike_dend_i + 0.06, spikes_post_i[0])

        axis.plot([spike_dend_i, spike_post_i], [i_neuron + y_offset, i_neuron + y_offset], color=color,
                  label=label)


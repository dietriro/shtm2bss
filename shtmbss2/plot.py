from matplotlib import pyplot as plt


def plot_membrane(membrane_recorded):
    membrane = membrane_recorded.get_data("v").segments[-1].irregularlysampledsignals[0]
    plt.plot(membrane.times, membrane, alpha=0.5)




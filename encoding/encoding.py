import pandas as pd
import numpy as np
from scipy.io import loadmat
from python_speech_features import logfbank
import matplotlib.pyplot as plt
import pickle
import os

data_paths = ["test","train"]
for path in data_paths:

    data = loadmat(f"{os.getcwd()}/data/TIDIGIT_{path}.mat")
    samples = data[f'{path}_samples']

    n_samples = len(samples)
    samplerate = 20000
    n_rows = 41
    n_frequency_bands = 40
    n_spiketime_bins = 30
    nfft = 1024

    def spectrogram(sample):
        """
        function to convert a signal to a spectrogram
        """
        duration = len(sample[0])/samplerate
        winstep = duration/n_rows
        fbank = logfbank(sample[0], samplerate=samplerate, winlen=1.1*
                        winstep, winstep=winstep, nfilt=n_frequency_bands, nfft=nfft)
        return fbank

    def spikepattern(spectrogram, bins):
        """
        function to convert spectrogram to (categorical) spikes, given the bins    
        """
        return np.digitize(spectrogram, bins)

    def spikepattern_local(spectrogram):
        max_amplitude = np.max(spectrogram)  # global max
        min_amplitude = np.min(spectrogram)  # global min
        bins = np.linspace(min_amplitude, max_amplitude, n_spiketime_bins + 1)  # bins are evenly spaced
        return np.digitize(spectrogram, bins)

    # Create spectrograms for each signal
    spectrograms = np.zeros((n_samples, n_rows, n_frequency_bands))
    for si, sample in enumerate(samples):
        if si%10 == 0:
            print(f"{si}th out of {n_samples} {path} samples \t spectrogram created")
        spect = spectrogram(sample)

        spectrograms[si] = spect

    # """
    # Construct the bins using global statistics. This is not made explicit in the paper,
    # and it is easily adjusted to only consider local statistics.
    # """
    # max_amplitude = np.max(spectrograms) # global max
    # min_amplitude = np.min(spectrograms) # global min
    # bins = np.linspace(min_amplitude, max_amplitude, n_spiketime_bins+1) # bins are evenly spaced

    # Create spike pattern for each spectrogram
    spike_patterns = np.zeros((n_samples, n_rows, n_frequency_bands))
    for si, spectrogram in enumerate(spectrograms):
        spike_patterns[si] = spikepattern_local(spectrogram)
        if si%10 == 0:
            print(f"{si}th out of {n_samples} {path} samples \t spike pattern created\r", end = "")
    pickle.dump(spike_patterns, open(f"data/ttfs_spikes_v2_{path}.p", "wb"))

# sample = samples[0, 0]
# spectrogram = spectrograms[0]
# spikepattern = spike_patterns[0][20]

# fig, ax = plt.subplots(1, 3)
# ax[0].plot(sample)
# ax[1].imshow(spectrogram.T)
# ax[2].imshow(spikepattern, cmap="Greys")
# plt.show()

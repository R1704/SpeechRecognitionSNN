import pandas as pd
import numpy as np
from scipy.io import loadmat
from python_speech_features import logfbank
import matplotlib.pyplot as plt

data = loadmat("data/TIDIGIT_test.mat")
samples = data['test_samples']

n_samples = len(samples)
samplerate = 20000
n_rows = 40
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
    function to convert spectrogram to spikes, given the bins    
    """
    sample_spikes = np.zeros((n_rows, n_frequency_bands, n_spiketime_bins))
    for ri, row in enumerate(spectrogram):
        row_spikes = pd.get_dummies(pd.cut(row, bins=bins))
        sample_spikes[ri] = row_spikes.to_numpy()
    return sample_spikes

# Create spectrograms for each signal
spectrograms = np.zeros((n_samples, n_rows, n_frequency_bands))
for si, sample in enumerate(samples):
    if si%10 == 0:
        print(f"{si}th out of {n_samples} \t spectrogram created\r", end = "")
    spect = spectrogram(sample)
    spectrograms[si] = spect

"""
Construct the bins using global statistics. This is not made explicit in the paper,
and it is easily adjusted to only consider local statistics.
"""
max_amplitude = np.max(spectrograms) # global max
min_amplitude = np.min(spectrograms) # global min
bins = np.linspace(min_amplitude, max_amplitude, n_spiketime_bins+1) # bins are evenly spaced

# Create spike pattern for each spectrogram
spike_patterns = np.zeros((n_samples, n_rows, n_frequency_bands, n_spiketime_bins))
for si, spectrogram in enumerate(spectrograms):
    if si%10 == 0:
        print(f"{si}th out of {n_samples} \t spike pattern created\r", end = "")
    spike_patterns[si] = spikepattern(spectrogram, bins)


# sample = samples[0, 0]
# spectrogram = spectrograms[0]
# spikepattern = spike_patterns[0][20]

# fig, ax = plt.subplots(1, 3)
# ax[0].plot(sample)
# ax[1].imshow(spectrogram.T)
# ax[2].imshow(spikepattern, cmap="Greys")
# plt.show()

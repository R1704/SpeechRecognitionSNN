import pandas as pd
from scipy.io import loadmat
import os
from python_speech_features import logfbank
import matplotlib.pyplot as plt
import numpy as np
import librosa
import librosa.display
import pickle
import math


sample_rate = 20000
n_fft = 2048
n_time_frames = 41
n_frequency_bands = 40
n_mels = 40
n_spiketime_bins = 31
hop_length = 512
threshold = 0.001  # to trim the signal


def spikepattern_local(spectrogram):
    max_amplitude = np.max(spectrogram)
    min_amplitude = np.min(spectrogram)
    bins = np.linspace(min_amplitude, max_amplitude, n_spiketime_bins + 1)  # bins are evenly spaced
    return np.digitize(spectrogram, bins)


def get_data(path, save=False):

    data = loadmat(f"{os.getcwd()}/data/TIDIGIT_{path}.mat")
    samples = data[f'{path}_samples']
    n_samples = len(samples)

    spectrograms = np.zeros((n_samples, n_time_frames, n_frequency_bands))
    for i, sample in enumerate(samples):
        if i % 10 == 0:
            print(f"{i}th out of {n_samples} {path} samples \t spectrogram created")

        sample = sample[0].flatten()

        # Trim and plot
        indx = np.where(abs(sample) > threshold)
        fig, ax = plt.subplots(2, 1)
        ax[0].plot(sample)
        sample = sample[indx]
        ax[1].plot(sample)
        plt.show()

        # Calculate hop length
        duration = sample.shape[0]
        hop_length = duration / n_time_frames
        hop_length = math.ceil(hop_length) if hop_length % 1 != 0 else math.ceil(hop_length+0.00000001)

        # Create Spectrogram from Mel bands
        mel = librosa.feature.melspectrogram(sample, sr=sample_rate, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels)
        spect = librosa.power_to_db(mel, ref=np.max)

        # Save spectogram
        spectrograms[i] = spect.T

        # Plot spectogram
        plt.imshow(spect.T)
        plt.show()

    spike_patterns = np.zeros((n_samples, n_time_frames, n_frequency_bands))
    for i, spect in enumerate(spectrograms):
        spike_patterns[i] = spikepattern_local(spect)
        if i % 10 == 0:
            print(f"{i}th out of {n_samples} {path} samples \t spike pattern created\r", end="")
    if save:
        pickle.dump(spike_patterns, open(f"ttfs_spikes_data/ttfs_spikes_mel_{path}.p", "wb"))

    return spike_patterns


def run():
    paths = ['train', 'test']
    results = []
    for path in paths:
        result = get_data(path)
        results.append(result)

    return results






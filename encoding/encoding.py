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
import glob
import regex as rgx
from pathlib import Path



n_time_frames = 41
n_frequency_bands = 40
n_mels = 40
n_spiketime_bins = 31
hop_length = 512
threshold = 0.001  # to trim the signal
a_little_bit = 0.00000001


def spikepattern_local(spectrogram):
    max_amplitude = np.max(spectrogram)
    min_amplitude = np.min(spectrogram)
    bins = np.linspace(min_amplitude, max_amplitude, n_spiketime_bins + 1)  # bins are evenly spaced
    return np.digitize(spectrogram, bins)

# def spectrogram(sample, sample_rate, n_fft):
#
#     # Calculate hop length
#     duration = sample.shape[0]
#     hop_length = duration / n_time_frames
#     hop_length = math.ceil(hop_length) if hop_length % 1 != 0 else math.ceil(hop_length + a_little_bit)
#
#     # Create Spectrogram from Mel bands
#     mel = librosa.feature.melspectrogram(sample, sr=sample_rate, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels)
#     spect = librosa.power_to_db(mel, ref=np.max)
#     return spect.T


samplerate = 20000
n_rows = 41
n_frequency_bands = 40
n_spiketime_bins = 30
nfft = 2048

def spectrogram(sample):
    """
    function to convert a signal to a spectrogram
    """
    duration = len(sample)/samplerate
    winstep = duration/n_rows
    fbank = logfbank(sample, samplerate=samplerate, winlen=1.1*
                    winstep, winstep=winstep, nfilt=n_frequency_bands, nfft=nfft)
    return fbank


def get_TIMIT(path, save=False):
    """
    This function goes through all directories of the TIMIT data set and creates a pickle which contains a dictionary
    of two lists. One list contains all labels and the other all spike patterns. Each sentence of the original
    data-set is chopped up into single words.
    Args:
        path: test or train
        save: makes a pickle if True

    Returns:
        the data
    """
    sample_rate = 16000
    n_fft = 512
    data = {f'{path}_samples': [], f'{path}_labels': []}
    words = ['that', 'she', 'all', 'your', 'me', 'had', 'like', "don't", 'year', 'water', 'dark', 'rag', 'oily', 'wash', 'ask', 'carry', 'suit']

    # Search through directories recursively
    for filename in glob.iglob(f'data/TIMIT/{path}' + '/**/*.WRD', recursive=True):
        wav_path = filename[:-3] + 'WAV'

        # Get the sentence and the waveform
        with open(filename, 'r') as f:
            sentence = f.read().split('\n')[:-1]
            audio, _ = librosa.load(wav_path, sr=sample_rate)

            # For each word in the sentence make a spike pattern and label
            for start_end_word in sentence:
                start, end, word = tuple(start_end_word.split(' '))

                if word in words:

                    if int(start) - int(end) != 0:

                        # Chop up sentence to single words
                        chopped = audio[int(start):int(end)]

                        # Show original vs chopped
                        fig, ax = plt.subplots(2, 1)
                        ax[0].plot(audio)
                        ax[1].plot(chopped)
                        plt.show()
                        print(start, end, word)

                        # Make the spectogram and spike pattern
                        spect = spectrogram(chopped, sample_rate, n_fft)
                        spike_pattern = spikepattern_local(spect)
                        if spect.shape == (41, 40):

                            # Show the spectogram
                            plt.imshow(spect)
                            plt.show()

                            # Save label
                            data[f'{path}_labels'].append(word)

                            # Save spike pattern
                            data[f'{path}_samples'].append(spike_pattern)
    if save:
        print(f'Saving TIMIT_mel_{path}')
        pickle.dump(data, open(f"ttfs_spikes_data/TIMIT_mel_v2_{path}.p", "wb"))

    return data


def get_TIDIGITS(path, save=False):
    sample_rate = 20000
    n_fft = 2048

    data = loadmat(f"{os.getcwd()}/data/TIDIGIT_{path}.mat")
    samples = data[f'{path}_samples']
    n_samples = len(samples)
    sample_avg_len = 0

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

        sample_avg_len += len(sample)

        # spect = spectrogram(sample, sample_rate, n_fft)
        spect = spectrogram(sample)

        if spect.shape == (41, 40):
            # Save spectrogram
            spectrograms[i] = spect

        # Plot spectrogram
        plt.imshow(spect)
        plt.show()

    spike_patterns = np.zeros((n_samples, n_time_frames, n_frequency_bands))
    for i, spect in enumerate(spectrograms):
        spike_patterns[i] = spikepattern_local(spect)
        if i % 10 == 0:
            print(f"{i}th out of {n_samples} {path} samples \t spike pattern created\r", end="")
    if save:
        pickle.dump(spike_patterns, open(f"ttfs_spikes_data/TIGIDITS_spikes_{path}.p", "wb"))

    print(sample_avg_len, n_samples, sample_avg_len/ n_samples)

    return spike_patterns


def run():
    paths = ['train', 'test']
    results = []
    for path in paths:
        print(f'starting with {path} data')
        result = get_TIDIGITS(path, save=False)
        results.append(result)

    return results
# run()









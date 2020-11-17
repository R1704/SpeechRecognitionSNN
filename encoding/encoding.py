import pandas as pd
import scipy as sp
from scipy.io import loadmat
from python_speech_features import logfbank

import matplotlib.pyplot as plt

data = loadmat("data/TIDIGIT_test.mat")
sample = data['test_samples'][0][0]

samplerate = 16000
duration = len(sample)/samplerate
number_of_rows = 41
n_frequency_bands = 40
winstep = duration/number_of_rows
fbank = logfbank(sample , samplerate = samplerate , winlen = 2 * winstep, winstep = winstep, nfilt = n_frequency_bands)


number_of_timesteps = 30
print(fbank)
# plt.imshow(fbank)
# plt.show()
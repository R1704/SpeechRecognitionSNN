import pandas as pd
import scipy as sp
from scipy.io import loadmat
from python_speech_features import logfbank

import matplotlib.pyplot as plt

data = loadmat("data/TIDIGIT_test.mat")
sample = data['test_samples'][1][0]

samplerate = 16000
duration = len(sample)/samplerate
number_of_rows = 100
winstep = duration/number_of_rows
fbank = logfbank(sample,samplerate=samplerate, winlen = 2*winstep, winstep = winstep)


# plt.imshow(fbank)
# plt.show()
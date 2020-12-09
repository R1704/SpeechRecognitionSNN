import numpy as np
import pandas as pd
from SpykeTorch import *
import torch.nn as nn
from SpykeTorch import snn
import SpykeTorch.functional as sf
import matplotlib.pyplot as plt
from torch import tensor
import torch
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


class Conv(nn.Module):
    def __init__(self):
        super(Conv, self).__init__()

        self.kernel_size = (4, 6)
        self.stride      = 1

        # in_channels=time-frames, out_channels=n_feature_maps
        self.conv = snn.Convolution(in_channels=40, out_channels=50, kernel_size=(4, 6)).double()

        # Stdp
        self.stdp1 = snn.STDP(conv_layer=self.conv, learning_rate=(0.004, -0.003))

        self.ctx = {"input_spikes": None, "potentials": None, "output_spikes": None, "winners": None}

        self.decision_map = []
        for i in range(10):
            self.decision_map.extend([i] * 20)


    def forward(self, x):
        pots = self.conv(x)
        spks = sf.fire(potentials=pots, threshold=15)  # not sure what the threshold should be yet
        pots = sf.pooling(spks, kernel_size=3)
        spks = sf.fire_(pots)

        # Get one winner and shut other neurons off; lateral inhibition
        winners = sf.get_k_winners(pots, 1)

        self.save_data(x, pots, spks, winners)

        output = -1

        if len(winners) != 0:
            output = self.decision_map[winners[0][0]]
        return output

    def save_data(self, inp_spks, pots, spks, winners):
        self.ctx['input_spikes'] = inp_spks
        self.ctx['potentials']   = pots
        self.ctx['spikes']       = spks
        self.ctx['winners']      = winners

    def stdp(self):
        self.stdp1(self.ctx['input_spikes'], self.ctx['potentials'], self.ctx['spikes'], self.ctx['winners'])




def read_data(filename):
    ttfs_spikes_train = pd.read_pickle(filename)
    return ttfs_spikes_train


def one_hot_encoding(data):
    """
    Computes one-hot encoding from data
    Returns: [samples, time-frames, frequency bands, time-points]
    """
    n_spiketime_bins = np.max(data).astype(int) + 1
    spikes = np.zeros(list(data.shape)+[n_spiketime_bins])
    for i, sample in enumerate(data):
        for j, tf in enumerate(sample):
            spikes[i, j] = np.eye(n_spiketime_bins)[tf]

    return spikes


def one_hot_decoding(data):
    # shape = data.shape
    # spikes = np.zeros((shape[0], shape[1], shape[2]))
    return data.argmax(axis=3)

def train(network, data):
    n_input_sections = 6  # 6 x n_frequency-bands
    n_feature_maps = 50
    n_conv_sections = 9
    n_section_length = 4

    network.train()

    outputs = np.zeros((n_conv_sections, 32, 50, 2461, 4))
    for i in range(n_conv_sections):
        # segment the data
        sec_data = data[:, i * n_section_length: i * n_section_length + (n_input_sections + n_section_length - 1), :, :]
        # print(f'shape of segmented data {sec_data.shape}')

        # reshape it to fit pytorch
        shape = sec_data.shape
        sec_data_r = np.reshape(sec_data, (shape[3], shape[2], shape[0], shape[1]))
        # print(f'reshaped segmented data {sec_data_r.shape}')

        # save output
        outputs[i] = network(tensor(sec_data_r).to(torch.double))
        network.stdp()
    return outputs

def run():

    # get data from file
    filename = r'ttfs_spikes_data/ttfs_spikes_train.p'
    ttfs_spikes_train = read_data(filename)
    print(f'original data shape (samples, timeframe, frequency) {ttfs_spikes_train.shape}')  # (sample, time-frame, frequency-band)

    # one hot encode to use on snn
    data_one_hot = one_hot_encoding(ttfs_spikes_train.astype(int))
    print(f'shape after one hot encoding {data_one_hot.shape}')  # [samples, time-frames, frequency-bands, time-points]

    # show example because we like visuals
    plt.imshow(ttfs_spikes_train[0])
    plt.show()

    # Initialise Convolutional layer
    network = Conv()

    # run model
    outputs = train(network, data_one_hot)
    print(outputs.shape)
    # reshape and show another example of a feature map
    ex = np.reshape(outputs, (2461, 9*4, 50, 32))
    ex_d = one_hot_decoding(ex)
    plt.imshow(ex_d[0])
    plt.show()

    # print(ex.shape)





if __name__ == '__main__':
    run()

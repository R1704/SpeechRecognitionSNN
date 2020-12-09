import numpy as np
import pandas as pd
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

        self.kernel_size = (6, 40)
        self.stride      = 1
        self.n_conv_sections = 9
        self.n_input_sections = 6  # 6 x 40 frequency-bands
        self.n_feature_maps = 50
        self.n_conv_sections = 9
        self.n_section_length = 4

        # in_channels=time-frames, out_channels=n_feature_maps

        self.convs = nn.ModuleList(
            [
                snn.Convolution(
                    in_channels=1, out_channels=50, kernel_size=self.kernel_size) for _ in range(self.n_conv_sections)
            ]
        )

        # Stdp
        self.stdps = nn.ModuleList(
            [snn.STDP(conv_layer=conv, learning_rate=(0.004, -0.003)) for conv in self.convs]
        )

        self.pools = nn.ModuleList(
            [snn.Pooling((self.n_section_length, self.n_section_length)) for _ in range(self.n_conv_sections)]
        )

        self.ctx = {"input_spikes": None, "potentials": None, "output_spikes": None, "winners": None}

        self.decision_map = []
        for i in range(10):
            self.decision_map.extend([i] * 20)

    def forward(self, x):


        outputs = []
        pools = []

        if not self.training:
            for i in range(self.n_conv_sections):
                # section the data
                sec_data = x[:, i * self.n_section_length: i * self.n_section_length + (self.n_input_sections + self.n_section_length - 1), :, :]

                # send each section through its convolutional layer
                pots = self.convs[i](sec_data)

                # get the spikes
                spks = sf.fire(potentials=pots, threshold=23)

                # pool data | in the paper they say pool by feature map and weight should be 1, is this correct?
                pots = self.pools[i](spks)
                pools.append(pots)

                # Get one winner and shut other neurons off; lateral inhibition
                winners = sf.get_k_winners(pots, 1, inhibition_radius=0)  # change inhibition radius ?
                output = -1

                if len(winners) != 0:
                    output = winners[0][0]
                    outputs.append(output)
                return output

        if self.training:

            for i in range(self.n_conv_sections):
                # section the data
                sec_data = x[:, i * self.n_section_length: i * self.n_section_length + (self.n_input_sections + self.n_section_length - 1), :, :]

                # send each section through its convolutional layer
                pots = self.convs[i](sec_data)

                # get the spikes
                spks = sf.fire(potentials=pots, threshold=23)

                # pool data | in the paper they say pool by feature map and weight should be 1, is this correct?
                pots = self.pools[i](spks)
                pools.append(pots)

                # Get one winner and shut other neurons off; lateral inhibition
                winners = sf.get_k_winners(pots, 1, inhibition_radius=0)  # change inhibition radius ?

                self.save_data(x, pots, spks, winners)

                output = -1

                if len(winners) != 0:
                    output = winners[0][0]
                    outputs.append(output)
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
    print(data.shape)
    network.train()
    # for d in data:
    #     network(d)
    outputs = network(tensor(data))
    print(outputs.shape)
    network.stdp()

    return outputs


    # outputs = np.zeros((n_conv_sections, 32, 50, 2461, 4))
    # for i in range(n_conv_sections):
    #
    #     # reshape it to fit pytorch
    #     shape = sec_data.shape
    #     sec_data_r = np.reshape(sec_data, (shape[3], shape[2], shape[0], shape[1]))
    #     # print(f'reshaped segmented data {sec_data_r.shape}')

        # save output
        # network.stdp()

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
    # ex = np.reshape(outputs, (2461, 9*4, 50, 32))
    # ex_d = one_hot_decoding(ex)
    # plt.imshow(ex_d[0])
    # plt.show()

    # print(ex.shape)





if __name__ == '__main__':
    run()

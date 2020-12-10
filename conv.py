import numpy as np
import pandas as pd
import scipy.io as sio
import torch.utils.data as data_utils
import torch.nn as nn
from SpykeTorch import snn
import SpykeTorch.functional as sf
import matplotlib.pyplot as plt
from torch import tensor
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
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
        self.n_frequency_bands = 40
        self.n_conv_sections = 9
        self.n_section_length = 4

        # in_channels=time-frames, out_channels=n_feature_maps

        self.convs = nn.ModuleList(
            [
                snn.Convolution(
                    in_channels=1, out_channels=50, kernel_size=self.kernel_size, weight_mean=0.8, weight_std=0.05
                ) for _ in range(self.n_conv_sections)
            ]
        )

        # Stdp
        self.stdps = nn.ModuleList(
            [snn.STDP(conv_layer=conv, learning_rate=(0.004, -0.003)) for conv in self.convs]
        )

        self.pools = nn.ModuleList(
            [snn.Pooling((self.n_section_length, self.n_frequency_bands)) for _ in range(self.n_conv_sections)]
        )

        self.ctx = {"input_spikes": None, "potentials": None, "output_spikes": None, "winners": None}


    def forward(self, x):

        if not self.training:
            # section the data
            sec_data = [x[:, :, i * self.n_section_length: i * self.n_section_length + (
                    self.n_input_sections + self.n_section_length - 1), :] for i in range(self.n_conv_sections)]

            # send each section through its convolutional layer
            pots = [self.convs[i](sec_data[i]) for i in range(self.n_conv_sections)]

            # get the spikes for each section
            spks = [sf.fire(potentials=pots[i], threshold=23) for i in range(self.n_conv_sections)]

            # pool the data
            pots = [self.pools[i](spks[i]) for i in range(self.n_conv_sections)]

            # Get one winner and shut other neurons off; lateral inhibition
            # change inhibition radius ?
            winners = [sf.get_k_winners(pots[i], 1, inhibition_radius=0) for i in range(self.n_conv_sections)]

            output = [-1 for _ in range(self.n_conv_sections)]

            for w in range(len(winners)):
                if len(winners[w]) != 0:
                    output[w] = winners[w][0]
            return output

        if self.training:


            # section the data by [9 tf x 40 fb]
            sec_data = [x[:, :, i * self.n_section_length: i * self.n_section_length + (
                        self.n_input_sections + self.n_section_length - 1), :] for i in range(self.n_conv_sections)]

            # send each section through its convolutional layer
            pots = [self.convs[i](sec_data[i]) for i in range(self.n_conv_sections)]

            # get the spikes for each section
            spks = [sf.fire(potentials=pots[i], threshold=23) for i in range(self.n_conv_sections)]

            # Get one winner and shut other neurons off; lateral inhibition
            # change inhibition radius ?
            winners = [sf.get_k_winners(pots[i], 1, inhibition_radius=0) for i in range(self.n_conv_sections)]

            self.save_data(sec_data, pots, spks, winners)

            output = [-1 for _ in range(self.n_conv_sections)]

            for w in range(len(winners)):
                if len(winners[w]) != 0:
                    output[w] = winners[w][0]
            return output

    def save_data(self, inp_spks, pots, spks, winners):
        self.ctx['input_spikes'] = inp_spks
        self.ctx['potentials']   = pots
        self.ctx['spikes']       = spks
        self.ctx['winners']      = winners

    def stdp(self):
        for i in range(len(self.stdps)):
            self.stdps[i](self.ctx['input_spikes'][i], self.ctx['potentials'][i], self.ctx['spikes'][i], self.ctx['winners'][i])


def train(network, data):
    network.train()
    for d in data:
        for x in d[0]:
            out = network(x.float())  # float?
            print(out)
            network.stdp()
    return

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

def prep_data():

    # load spikes (41 frames, 40 frequency bands)
    ttfs_spikes_train = pd.read_pickle(r'ttfs_spikes_data/ttfs_spikes_train.p')
    ttfs_spikes_test = pd.read_pickle(r'ttfs_spikes_data/ttfs_spikes_test.p')
    ttfs_spikes_all = np.concatenate((ttfs_spikes_train, ttfs_spikes_test), axis=0)

    # one hot encode to use on snn
    spikes = one_hot_encoding(ttfs_spikes_all.astype(int))
    print(f'shape after one hot encoding {spikes.shape}')  # [samples, time-frames, frequency-bands, time-points]

    # show example because we like visuals
    # plt.imshow(ttfs_spikes_train[0])
    # plt.show()

    # switch axes because spyketorch
    spikes = np.swapaxes(spikes, 1, 3)
    print(f'shape after switching axes {spikes.shape}')  # [samples, time-points, frequency-bands, time-frames]

    # add channel dimension [samples, time-points, channels, time-frames, frequency bands]
    # (samples, 32, 1, 40, 41)
    spikes = spikes[:, :, np.newaxis, :]
    print(spikes.shape)

    # load labels
    train_mat = sio.loadmat("data/TIDIGIT_train.mat")
    test_mat = sio.loadmat("data/TIDIGIT_test.mat")
    train_targets = train_mat['train_labels'].astype(int)
    test_targets = test_mat['test_labels'].astype(int)
    all_targets = np.concatenate((train_targets, test_targets), axis=0)

    # split data
    X_train, X_test, y_train, y_test = train_test_split(spikes, all_targets, test_size=0.3, random_state=42)
    print("X_train", X_train.shape, "X_test", X_test.shape)

    # prepare Dataloader
    train_torchset = data_utils.TensorDataset(torch.tensor(X_train), torch.tensor(y_train))
    test_torchset = data_utils.TensorDataset(torch.tensor(X_test), torch.tensor(y_test))
    train_loader = DataLoader(train_torchset, batch_size=64)
    test_loader = DataLoader(test_torchset, batch_size=64)

    return train_loader, test_loader


def run():

    # Prepare and get data
    train_loader, test_loader = prep_data()

    # Initialise Convolutional layer
    network = Conv()

    # run model
    output = train(network, train_loader)


    print(output)
    # reshape and show another example of a feature map
    # ex = np.reshape(outputs, (2461, 9*4, 50, 32))
    # ex_d = one_hot_decoding(ex)
    # plt.imshow(ex_d[0])
    # plt.show()

    # print(ex.shape)





if __name__ == '__main__':
    run()

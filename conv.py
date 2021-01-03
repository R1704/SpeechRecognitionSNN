import numpy as np
import pandas as pd
import scipy.io as sio
import torch.utils.data as data_utils
import torch.nn as nn
from SpykeTorch import snn
import SpykeTorch.functional as sf
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import os
import encoding
from sklearn.preprocessing import StandardScaler
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


class Conv(nn.Module):
    def __init__(self):
        super(Conv, self).__init__()

        self.kernel_size       = (6, 40)
        self.stride            = 1
        self.n_conv_sections   = 9
        self.n_input_sections  = 6
        self.n_feature_maps    = 50
        self.n_frequency_bands = 40
        self.n_conv_sections   = 9
        self.n_section_length  = 4

        self.threshold = 6.2


        # Convolution
        self.convs = nn.ModuleList(
            [
                snn.Convolution(
                    in_channels=1, out_channels=50, kernel_size=self.kernel_size, weight_mean=0.8, weight_std=0.05
                ) for _ in range(self.n_conv_sections)
            ]
        )

        # STDP
        self.stdps = nn.ModuleList(
            [snn.STDP(conv_layer=conv, learning_rate=(0.004, -0.003)) for conv in self.convs]
        )

        # Pooling
        self.pools = nn.ModuleList(
            [snn.Pooling((4, 1)) for _ in range(self.n_conv_sections)]  # not sure whether the pooling kernel is right.
        )

        self.ctx = {"input_spikes": None, "potentials": None, "output_spikes": None, "winners": None}

    def forward(self, x):

        if self.training:
            # Reset lists for STDP
            self.ctx = {"input_spikes": None, "potentials": None, "output_spikes": None, "winners": None}
            # section the data by [9 tf x 40 fb]
            sec_data = [x[:, :, i * self.n_section_length: i * self.n_section_length + (self.n_input_sections + self.n_section_length - 1), :] for i in range(self.n_conv_sections)]
            # send section through its convolutional layer
            pots = [conv(sec_data[i]) for i, conv in enumerate(self.convs)]  # shape = [32, 50, 4, 1]
            # get the spikes for each section
            spks = [sf.fire(potentials=pot, threshold=self.threshold) for pot in pots]
            # Get one winner and shut other neurons off; lateral inhibition
            winners = [sf.get_k_winners(pots[i], 1, inhibition_radius=0, spikes=spks[i]) for i in range(self.n_conv_sections)]  # change inhibition radius ?
            self.save_data(sec_data, pots, spks, winners)

        if not self.training:
            # section the data by [9 tf x 40 fb] ==> shape = [32, 1, 9, 40]
            sec_data = [x[:, :, i * self.n_section_length: i * self.n_section_length + (self.n_input_sections + self.n_section_length - 1), :] for i in range(self.n_conv_sections)]
            # send section through its convolutional layer
            pots = [conv(sec_data[i]) for i, conv in enumerate(self.convs)]  # shape = [32, 50, 4, 1]
            # pool each section
            pots = [pool(pots[i]) for i, pool in enumerate(self.pools)]  # pots.shape [32, 50, 1, 1]
            # get the spikes for each section
            spks = [sf.fire(potentials=pot, threshold=self.threshold) for pot in pots]
            # Put all pools together
            pots = torch.cat(pots, dim=2)  # shape = [32, 50, 9, 1]
            spks = torch.cat(spks, dim=2)

            return pots, spks

    def save_data(self, inp_spks, pots, spks, winners):
        self.ctx['input_spikes'] = inp_spks
        self.ctx['potentials'] = pots
        self.ctx['output_spikes'] = spks
        self.ctx['winners'] = winners

    def stdp(self):
        for i, stdp in enumerate(self.stdps):
            stdp(self.ctx['input_spikes'][i], self.ctx['potentials'][i], self.ctx['output_spikes'][i], self.ctx['winners'][i])


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

    # Creating data
    # results = encoding.run()
    # ttfs_spikes_train = results[0]
    # ttfs_spikes_test = results[1]

    # Loading data (41 frames, 40 frequency bands)
    #ttfs_spikes_train = pd.read_pickle(r'ttfs_spikes_data/ttfs_spikes_v2_train.p')
    #ttfs_spikes_test = pd.read_pickle(r'ttfs_spikes_data/ttfs_spikes_v2_test.p')
    ttfs_spikes_train = pd.read_pickle(r'ttfs_spikes_data/ttfs_spikes_mel_train.p')
    ttfs_spikes_test = pd.read_pickle(r'ttfs_spikes_data/ttfs_spikes_mel_test.p')
    ttfs_spikes_all = np.concatenate((ttfs_spikes_train, ttfs_spikes_test), axis=0)

    # one hot encode to use on snn
    spikes = one_hot_encoding(ttfs_spikes_all.astype(int))
    print(f'shape after one hot encoding {spikes.shape}')  # [samples, time-frames, frequency-bands, time-points] [4950, 41, 40, 32]

    # show example because we like visuals
    plt.imshow(ttfs_spikes_train[0])
    plt.show()

    # switch axes because spyketorch
    spikes = np.reshape(spikes, (spikes.shape[0], spikes.shape[3], spikes.shape[1], spikes.shape[2]))
    print(f'shape after reshaping {spikes.shape}')  # [samples, time-points, time-frames, frequency-bands]

    # add channel dimension [samples, time-points, channels, time-frames, frequency bands]
    # (samples, 32, 1, 41, 40)
    spikes = spikes[:, :, np.newaxis, :]
    print(f'shape after adding axis {spikes.shape}')

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


def train(network, data, n_epochs=1):
    print('Starting training ...')
    network.train()
    for e in range(n_epochs):
        print(f'Starting epoch {e+1}')
        for d, _ in data:
            for x in d:
                network(x.float())
                network.stdp()
    print('Training done ...')


def evaluation(network, loader):
    ypots = []  # outputs (pots)
    yspikes = []
    ts = []  # targets
    print('Starting evaluation ...')
    network.eval()
    for data, targets in loader:
        for x, t in zip(data, targets):
            pots, spikes = network(x.float())
            ypots.append(pots.reshape(-1).cpu().numpy())
            yspikes.append(spikes.reshape(-1).cpu().numpy())
            ts.append(t)
    print('Evaluation done ...')
    return np.asarray(ypots), np.asarray(yspikes), np.asarray(ts)


def classify(ys_train, ts_train, ys_test, ts_test, iterations=1000):
    # Fit classifier
    svc = LinearSVC(verbose=0, dual=False, max_iter=iterations, C=1)
    svc.fit(ys_train, ts_train)

    # Inference
    pred_train = svc.predict(ys_train)
    pred_test = svc.predict(ys_test)
    print(f'SVC run with {iterations} iterations')
    print(f'Accuracy on training data: {accuracy_score(ts_train, pred_train)}')
    print(f'Accuracy on testing data: {accuracy_score(ts_test, pred_test)}')


def run():

    # Prepare and get data
    train_loader, test_loader = prep_data()

    # Initialise Convolutional layer
    network = Conv()

    # run model
    train(network, train_loader, n_epochs=1)

    # Evaluate
    ypots_train, yspikes_train, ts_train = evaluation(network, train_loader)
    ypots_test, yspikes_test, ts_test = evaluation(network, test_loader)

    # Classify
    print("classify potential")
    classify(ypots_train, ts_train, ypots_test, ts_test, iterations=2500)
    print("classify spikes")
    classify(yspikes_train, ts_train, yspikes_test, ts_test, iterations=2500)

    # TODO: show what happens in feature maps

    # Results:
    # accuracy at 0.84 with 2500 (and 0.85 with 5000) classifier iterations. Plain version

    # Trimming and librosa's melspectogram converted to spectogram and then to spikepattern (2500 iterarions)
    # Accuracy on training data: 1.0
    # Accuracy on testing data: 0.9387205387205387

    # Librosa's melspectogram converted to spectogram and then to spikepattern (no trimming) (2500 iterarions)
    # Accuracy on training data: 0.9971139971139971
    # Accuracy on testing data: 0.804040404040404

    # Original encoding but trimmed
    # SVC run with 2500 iterations
    # Accuracy on training data: 1.0
    # Accuracy on testing data: 0.9387205387205387

    # Solved convergence warning & added spikes classifier
    # classify potential
    # SVC run with 2500 iterations
    # Accuracy on training data: 1.0
    # Accuracy on testing data: 0.9508417508417508
    # classify spikes
    # SVC run with 2500 iterations
    # Accuracy on training data: 1.0
    # Accuracy on testing data: 0.960942760942761

if __name__ == '__main__':
    run()

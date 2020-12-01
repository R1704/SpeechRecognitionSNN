import pandas as pd
import numpy as np
import scipy.io as sio
from SpykeTorch import snn as snn
from SpykeTorch import functional as sf
import torch.utils.data as data_utils
import torch
from torch.utils.data import DataLoader
from sklearn.svm import LinearSVC

ttfs_spikes_train = pd.read_pickle(r'ttfs_spikes_data/ttfs_spikes_train.p')
ttfs_spikes_test = pd.read_pickle(r'ttfs_spikes_data/ttfs_spikes_test.p')

def one_hot2spikes(data):
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

train = one_hot2spikes(ttfs_spikes_train.astype(int))
test = one_hot2spikes(ttfs_spikes_test.astype(int))

# [samples, time-frames, frequency bands, time-points] to [samples, time-points, time-frames, frequency bands]
train = np.swapaxes(train, 1, 3)
test = np.swapaxes(test, 1, 3)

# add channel dimension
train = train[:, :, np.newaxis, :]
test = test[:, :, np.newaxis, :]

# train labels
train_mat = sio.loadmat("data/TIDIGIT_train.mat")
test_mat = sio.loadmat("data/TIDIGIT_test.mat")

train_targets = train_mat['train_labels'].astype(int)
test_targets = test_mat['test_labels'].astype(int)

# prepare Dataloader
train_torchset = data_utils.TensorDataset(torch.tensor(train), torch.tensor(train_targets))
test_torchset = data_utils.TensorDataset(torch.tensor(test), torch.tensor(test_targets))

train_loader = DataLoader(train_torchset, batch_size=1)
test_loader = DataLoader(test_torchset)



pool = snn.Pooling(kernel_size = (1,4), stride = 2)
conv = snn.Convolution(in_channels=1, out_channels=32, kernel_size=(40,6))
stdp = snn.STDP(conv_layer = conv, learning_rate = (0.05, -0.015))

print("Starting Unsupervised Training ...")
for iter in range(1):
    print('\rIteration:', iter, end="")
    for data in train_loader:
        for x in data[0]:
            p = conv(x.float())
            o, p = sf.fire(p, 20, return_thresholded_potentials=True)
            winners = sf.get_k_winners(p, kwta=1, inhibition_radius=0, spikes=o)
            stdp(x, p, o, winners)
            x = pool(p)
print()
print("Unsupervised Training is Done.")


# Evaluation

# Training Vectors
train_x_spike = []
train_x_pot = []
train_y = []
for data,targets in train_loader:
    for x,t in zip(data, targets):
        p = conv(x.float())
        o = sf.fire(p, 20)
        x = pool(p) # before sf.fire?
        train_x_spike.append(o.reshape(-1).cpu().numpy())
        train_x_pot.append(p.reshape(-1).cpu().numpy())
        train_y.append(t)
train_x_spike = np.array(train_x_spike)
train_x_pot = np.array(train_x_pot)
train_y = np.array(train_y)

# Testing Vectors
test_x_spike = []
test_x_pot = []
test_y = []
for data,targets in test_loader:
    for x,t in zip(data, targets):
        p = conv(x.float())
        o = sf.fire(p, 20)
        x = pool(p)
        test_x_spike.append(o.reshape(-1).cpu().numpy())
        test_x_pot.append(p.reshape(-1).cpu().numpy())
        test_y.append(t)
test_x_spike = np.array(test_x_spike)
test_x_pot = np.array(test_x_pot)
test_y = np.array(test_y)

# Classifier
clf_spike = LinearSVC(max_iter=10) #100000
clf_pot = LinearSVC(max_iter=10) #100000
clf_spike.fit(train_x_spike, train_y)
clf_pot.fit(train_x_pot, train_y)

# Inference
predict_spike = clf_spike.predict(test_x_spike)
predict_pot = clf_pot.predict(test_x_pot)

error_spike = np.abs(test_y - predict_spike).sum()
error_pot = np.abs(test_y - predict_pot).sum()
print("    Spike-based error:", error_spike/len(predict_spike))
print("Potential-based error:", error_pot/len(predict_pot))
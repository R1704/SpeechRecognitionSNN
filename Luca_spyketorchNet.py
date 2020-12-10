import pandas as pd
import numpy as np
import scipy.io as sio
from SpykeTorch import snn as snn
from SpykeTorch import functional as sf
import torch.utils.data as data_utils
import torch
from torch.utils.data import DataLoader
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

epochs = 1
clf_iter = 50

# load spikes (41 frames, 40 frequency bands)
ttfs_spikes_train = pd.read_pickle(r'ttfs_spikes_data/ttfs_spikes_train.p')
ttfs_spikes_test = pd.read_pickle(r'ttfs_spikes_data/ttfs_spikes_test.p')
ttfs_spikes_all = np.concatenate((ttfs_spikes_train, ttfs_spikes_test), axis=0)

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

spikes = one_hot2spikes(ttfs_spikes_all.astype(int))
# [samples, time-frames, frequency bands, time-points] to [samples, time-points, frequency bands, time-frames]
# (samples, 41, 40, 32) to (samples, 32, 40, 41)
spikes = np.swapaxes(spikes, 1, 3)
print(spikes.shape)

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

# train network
pool = snn.Pooling(kernel_size = (4,1), stride = 1)
conv = snn.Convolution(in_channels=1, out_channels=10, kernel_size=(6,40), weight_mean=0.8, weight_std=0.05)
stdp = snn.STDP(conv_layer = conv, learning_rate = (0.004, -0.003))

print("Starting Unsupervised Training ...")
for iter in range(epochs):
    print('\rIteration:', iter, end="")
    for data in train_loader:
        for x in data[0]:
            #print("x.shape", x.shape)
            p = conv(x.float())
            #print("p.shape after conv",p.shape)
            p = pool(p)  # x = pool(p)
            o, p = sf.fire(p, 23, return_thresholded_potentials=True)
            winners = sf.get_k_winners(p, kwta=1, inhibition_radius=0, spikes=o)
            stdp(x, p, o, winners)

            #print("final", x.shape)
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
        p = pool(p)
        o = sf.fire(p, 23)
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
        p = pool(p)
        o = sf.fire(p, 23)
        test_x_spike.append(o.reshape(-1).cpu().numpy())
        test_x_pot.append(p.reshape(-1).cpu().numpy())
        test_y.append(t)
test_x_spike = np.array(test_x_spike)
test_x_pot = np.array(test_x_pot)
test_y = np.array(test_y)

# Classifier
clf_spike = LinearSVC(max_iter=clf_iter, verbose=1)
clf_pot = LinearSVC(max_iter=clf_iter,verbose=1)
clf_spike.fit(train_x_spike, train_y)
clf_pot.fit(train_x_pot, train_y)

# Inference
predict_spike = clf_spike.predict(test_x_spike)
predict_pot = clf_pot.predict(test_x_pot)
print("epochs", epochs)
print("clf_iter", clf_iter)
print("accuracy predict spike", accuracy_score(y_test, predict_spike))
print("accuracy predict pot", accuracy_score(y_test, predict_pot))

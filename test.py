import bindsnet as bn
import torch
import pickle
from playground_folder.convnet import Convnet

def data_to_one_hot(data, bins=30):
    eye = torch.eye(bins)
    onehot = torch.zeros(bins, 1, *data.shape)
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            onehot[:,0,i,j] = eye[int(data[i,j])]
    return onehot
data = pickle.load(open("ttfs_spikes_data/ttfs_spikes_test.p", "rb"))
input_size, n_feature_maps, kernel_size,  n_sections, section_length_postsyn = (41, 40), 50, (6, 40),  9, 4
net = Convnet(input_size, n_feature_maps, kernel_size,  n_sections, section_length_postsyn)

net.train()
for i in range(len(data)):
    data_in = data_to_one_hot(data[i])
    net(data_in)
    net.stdp()
import bindsnet as bn
from playground_folder.convnet import Convnet

input_size, n_feature_maps, kernel_size,  n_sections, section_length_postsyn = (41, 40), 50, (6, 40),  9, 4
net = Convnet(input_size, n_feature_maps, kernel_size,  n_sections, section_length_postsyn)

net.train()

for i in range(len(data)):
    data_in = data[i]
    net(data_in)
    net.stdp()
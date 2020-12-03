import bindsnet as bn
from playground_folder.convnet import Convnet

input_size, n_feature_maps, kernel_size, stride, n_sections, section_length = (41,40), 50,(6,40),1,9,4
net = Convnet(input_size, n_feature_maps, kernel_size, stride, n_sections, section_length)
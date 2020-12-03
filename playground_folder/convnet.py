from SpykeTorch.functional import *
from SpykeTorch.snn import *
import torch

class Convnet:
    def __init__(self, input_size, n_feature_maps, kernel_size, stride, n_sections,section_length):
        self.n_timesteps, self.frequency_bands = input_size
        self.n_feature_maps = n_feature_maps
        self.kernel_size = kernel_size
        self.stride = stride
        self.n_sections = n_sections
        self.section_length = section_length

        self.overlap = ((self.kernel_size * self.n_sections) - self.n_timesteps)/(n_sections-1)

        print(self.overlap)

from SpykeTorch.functional import *
from SpykeTorch.snn import *
import torch
import torch.nn

class Convnet(nn.Module):
    def __init__(self, input_size, n_feature_maps, kernel_size, n_sections,section_length_postsyn,n_channels):
        super(Convnet, self).__init__()
        self.n_timesteps, self.frequency_bands = input_size
        self.kernel_timesize,self.kernel_frequencysize = kernel_size
        self.n_feature_maps = n_feature_maps
        self.kernel_size = kernel_size
        self.n_sections = n_sections
        self.section_length_postsyn = section_length_postsyn
        self.section_length_presyn = self.kernel_timesize+section_length_postsyn-1
        self.n_channels = n_channels
        self.overlap = ((self.section_length_presyn * self.n_sections) - self.n_timesteps)/(n_sections-1)
        self.section_distance = self.section_length_presyn - self.overlap
        self.convlayers = torch.nn.ModuleList([Convolution(1,self.n_channels,(self.kernel_timesize,self.kernel_frequencysize)) for c in range(n_sections)])
        self.stdps = torch.nn.ModuleList([STDP(i, (0.004, -0.003)) for i in self.convlayers])
        self.poollayers = torch.nn.ModuleList([Pooling((self.section_length, self.kernel_frequencysize)) for c in range(n_sections)])


    def forward(self,X):
        if not self.training:
            X_sections = [X[(i*self.section_distance):(i*self.section_distance)+self.section_length_presyn] for i in range(self.n_sections)]
            section_pots = [self.convlayers[i](X_sections[i]) for i in range(self.n_sections)]
            spikes = [sf.fire(section_pots[i], 15) for i in range(self.n_sections)]
            pools = [self.poollayers[i](spikes[i]) for i in range(self.n_sections)]
            pool = torch.stack(pools,dim=0)
            winners = sf.get_k_winners(pool)
            output = -1
            if len(winners)!=0:
                output = winners[0][0]

            return output



    def save_data(self, input_spk, pot, spk, winners):
        self.ctx["input_spikes"] = input_spk
        self.ctx["potentials"] = pot
        self.ctx["output_spikes"] = spk
        self.ctx["winners"] = winners


from SpykeTorch.functional import *
from SpykeTorch.snn import *
import torch
import torch.nn

class Convnet(nn.Module):
    def __init__(self, input_size, n_feature_maps, kernel_size, n_sections,section_length_postsyn):
        super(Convnet, self).__init__()
        self.n_timesteps, self.frequency_bands          = input_size
        self.kernel_timesize,self.kernel_frequencysize  = kernel_size
        self.n_feature_maps                             = n_feature_maps
        self.kernel_size                                = kernel_size
        self.n_sections                                 = n_sections
        self.section_length_postsyn                     = section_length_postsyn
        self.section_length_presyn                      = self.kernel_timesize+section_length_postsyn-1
        self.overlap                                    = ((self.section_length_presyn * self.n_sections) - self.n_timesteps)/(n_sections-1)
        self.section_distance                           = int(self.section_length_presyn - self.overlap)

        self.convlayers = torch.nn.ModuleList([Convolution(1,self.n_feature_maps,(self.kernel_timesize,self.kernel_frequencysize)) for c in range(n_sections)])

        self.stdps = torch.nn.ModuleList([STDP(i, (0.004, -0.003)) for i in self.convlayers])

        self.poollayers = torch.nn.ModuleList([Pooling((self.section_length_postsyn, self.kernel_frequencysize)) for c in range(n_sections)])

        self.ctx = {}

    def forward(self,X):
        if not self.training:
            X_sections      = [X[:,0,(i*self.section_distance):(i*self.section_distance)+self.section_length_presyn] for i in range(self.n_sections)]
            section_pots    = [self.convlayers[i](X_sections[i]) for i in range(self.n_sections)]
            spikes          = [sf.fire(section_pots[i], 15,True) for i in range(self.n_sections)]
            pools           = [self.poollayers[i](spikes[i]) for i in range(self.n_sections)]
            pool            = torch.stack(pools,dim=0)
            winners         = sf.get_k_winners(pool)
            output          = -1
            if len(winners)!=0:
                output = winners[0][0]
            return output, pool
        elif self.training:
            X_sections      = [X[:,0,(i*self.section_distance):(i*self.section_distance)+self.section_length_presyn] for i in range(self.n_sections)]
            X_sections = [X[:,None] for X in X_sections]
            section_pots    = [self.convlayers[i](X_sections[i]) for i in range(self.n_sections)]
            spikes_pots     = [sf.fire(section_pots[i], 1, True) for i in range(self.n_sections)]
            winners         = [sf.get_k_winners(sp[1]) for sp in spikes_pots]
            self.save_data(X_sections, [sp[0] for sp in spikes_pots], [sp[1] for sp in spikes_pots], winners)
            output          = self.n_sections*[-1]
            for s in range(self.n_sections):
                if len(winners[s])!=0:
                    output[s] = winners[s][0]
            return output



    def save_data(self, input_spk, pot, spk, winners):
        self.ctx["input_spikes"] = input_spk
        self.ctx["potentials"] = pot
        self.ctx["output_spikes"] = spk
        self.ctx["winners"] = winners

    def stdp(self):
        input_spks= self.ctx["input_spikes"]
        pots = self.ctx["potentials"]
        spks= self.ctx["output_spikes"]
        winnerss = self.ctx["winners"]
        print(sum([len(w) > 0 for w in winnerss]))
        for i in range(len(pots)):
            self.stdps[i](input_spks[i],pots[i],spks[i],winnerss[i])




import os
import numpy as np
import torch
from torch.utils import model_zoo
import numpy as np
import matplotlib.pyplot as plt
from torch.nn import Module, Sequential
from collections import OrderedDict
import neur.cornet_v as cornet
# from rlpyt.utils.collections import namedarraytuple


class NeuronalModel:
    def __init__(self, input_dim):  # input_dim includes chan
        self.cornet = None
        self.recordings = None
        self.recordings_i = None
        self.stims = None
        self.layers = ('V1', 'V2', 'V4', 'IT')
        self.states = {l: [] for l in self.layers}  # look out, refs to the arrays must be kept constant

        self.reset(hard=True)

        # build mapping between neuron indices and their corresponding sites
        self.site_mapping = []
        self.layer_dims = OrderedDict([('V1', (64, input_dim[1] // 4, input_dim[2] // 4)), ('V2', (128, input_dim[1] // 8, input_dim[2] // 8)),
                                       ('V4', (256, input_dim[1] // 16, input_dim[2] // 16)), ('IT', (512, input_dim[1] // 32, input_dim[2] // 32))])
        for layer in self.layers:
            nchan, height, width = self.layer_dims[layer]
            self.site_mapping.extend([(layer, (c, h, w)) for c in range(nchan) for h in range(height) for w in range(width)])

        self.Gd = {s: i for i, s in enumerate(self.site_mapping)}  # inverse of self.site_mapping
        self.site_mapping = np.array([[m[0], np.array(m[1])] for m in self.site_mapping])  # now we can turn site mapping to np array for faster slicing

    def reset(self, hard=False):
        for a in self.states.values():  # just clear, keep list references constant
            a.clear()
        self.recordings = {}
        self.stims = {}

        if hard:  # reset
            self.cornet = cornet.cornet_v(pretrained=True)

    class OutputReader:
        def __init__(self, state_arr):
            self.state_arr = state_arr

        def __call__(self, module, input, output):
            self.state_arr.append(output)

    def build(self, Gndi):
        hooked_layers = set()
        for n, d in Gndi.items():
            layer = self.site_mapping[n][0]
            if layer not in hooked_layers:
                getattr(self.cornet.module, layer).register_forward_hook(NeuronalModel.OutputReader(self.states[layer]))
                hooked_layers.add(layer)  # add hook for whole layer

        return Gndi

    def record(self, Gndi, sites):
        # sites: nsteps x batch x nsites, where nsites is the number of sites that can be recorded simultaneously
        # generate recordings_i: nsteps x {layers_i: (batch_i, site_i, slice(batch_i, chan_i, h_i, w_i))}
        #   batch_i is an array of indices, helps rebuild the recorded signals into nsteps x batch x nsites
        #   same for site_i, indicates the index at which the recording was commanded by the control model
        self.recordings = []  # contains the actual recordings, filled in sim()
        self.recordings_i = []  # helper array, prepared for easy layer slicing in sim()
        _, _, self.rec_nsites = sites.shape

        for step in sites:
            self.recordings_i.append({})
            ptrs = self.site_mapping[step.flatten()].reshape(step.shape + (-1,))  # np array of pointers to the neurons
            for layer in self.layers:
                batch_i, site_i = np.where(ptrs[:, :, 0] == layer)
                if len(batch_i) > 0:
                    # rec_i indices the layer dim-by-dim, each dim is prepared to be sliced with an array
                    rec_i = np.concatenate([[batch_i], np.stack(ptrs[batch_i, site_i, 1]).transpose()])
                    self.recordings_i[-1][layer] = [batch_i, site_i, rec_i]

        return Gndi

    def stim(self, Gndi, sites, signals):
        # sites: nsteps x batch x nsites, where nsites is the number of sites that can be stimulated simultaneously
        # signals: nsteps x batch x nsites
        # generated stims vector: nsteps x {layer: batch x nchannels x height x width}
        self.stims = []
        _, batch_size, _ = sites.shape

        for step_site, step_signal in zip(sites, signals):
            self.stims.append({})
            ptrs = self.site_mapping[step_site.flatten()].reshape(step_site.shape + (-1,))  # np array of pointers to the neurons
            for layer in self.layers:
                batch_i, site_i = np.where(ptrs[:, :, 0] == layer)
                stim_vec = torch.zeros((batch_size,) + self.layer_dims[layer]).to('cuda')

                if len(batch_i) > 0:
                    # stim_i indexes stim_vec like:
                    #   [batch_0, batch_1, ..., batch_n], [chan_0, chan_1, ..., chan_n], [h_0, ...], [w_0, ...]
                    stim_i = np.concatenate([[batch_i], np.stack(ptrs[batch_i, site_i, 1]).transpose()])
                    stim_vec[stim_i] = step_signal[batch_i, site_i]

                self.stims[-1][layer] = stim_vec.to('cuda')

        return Gndi

    def sim(self, inputs, nsteps):
        outputs = []

        # simulation, recording and stimulation "in one line"
        for a in self.states.values():  # just clear, keep list references constant
            a.clear()  # states are filled when running cornet
        for s in range(nsteps):
            out = self.cornet(inputs[s % len(inputs)], self.stims[s])
            outputs.append(out)

        # retrieving states = recording
        self.recordings = torch.zeros((nsteps, inputs.shape[1], self.rec_nsites)).to('cuda')
        for step_i, rec in enumerate(self.recordings_i):  # recordings should be nsteps long
            for layer, layer_rec in rec.items():
                batch_i, site_i, rec_i = layer_rec
                self.recordings[step_i, batch_i, site_i] = self.states[layer][step_i][rec_i]

        return self.recordings, outputs


if __name__ == '__main__':
    input_dim = (3, 64, 64)
    neur = NeuronalModel(input_dim)
    Gndi = {s: i for i, s in enumerate(np.cumsum([d[0] * d[1] - 1 for _, d in neur.layer_dims.items()]))}
    neur.build(Gndi)  # ({'V1': 0, 'V2': 1, 'V4': 2, 'IT': 3})

    nsteps = 1
    batch_size = 2
    nsites = 3

    sites = torch.zeros((nsteps, batch_size, nsites), dtype=int).numpy()
    signals = torch.zeros((nsteps, batch_size, nsites)).to('cuda') + 4
    inputs = torch.zeros((nsteps, batch_size) + input_dim).to('cuda')

    neur.record(Gndi, sites)
    neur.stim(Gndi, sites, signals)
    neur.sim(inputs, nsteps)

    print(neur.states['V1'][0].shape)
    print(neur.states['V2'][0].shape)
    print(neur.states['V4'][0].shape)
    print(neur.states['IT'][0].shape)

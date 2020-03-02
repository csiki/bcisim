import os
import numpy as np
import torch
from torch.utils import model_zoo
import cornet
import numpy as np
import matplotlib.pyplot as plt
from torch.nn import Module, Sequential
from collections import OrderedDict


# TODO move updated "cornet_t" to this repo, add extra input layers to it that stims that intermediate layers;
#   in cornet_t.forward just slice the input to separate the inputs to different layers


class NeuronalModel:
    def __init__(self, input_dim):
        self.cornet = None
        self.states = None
        self.recordings = None
        self.stims = None
        self.reset()

        # build mapping between neuron indices and their corresponding sites
        self.site_mapping = []
        self.conv_dims = OrderedDict([('V1', (input_dim[0] // 4, input_dim[1] // 4)), ('V2', (input_dim[0] // 8, input_dim[1] // 8)),
                                      ('V4', (input_dim[0] // 16, input_dim[1] // 16)), ('IT', (input_dim[0] // 32, input_dim[1] // 32))])
        for layer in ('V1', 'V2', 'V4', 'IT'):
            d = self.conv_dims[layer]
            self.site_mapping.extend([(layer, (i, j)) for i in range(d[0]) for j in range(d[1])])
        self.Gd = np.arange(len(self.site_mapping))

    def reset(self):
        self.cornet = cornet.cornet_t(pretrained=True)
        self.states = {'V1': [], 'V2': [], 'V4': [], 'IT': []}
        self.recordings = {}
        self.stims = {}

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

    def record(self, Gndi, sites, nsteps):
        self.recordings = {s: [0] * nsteps for s in sites}
        return Gndi

    def stim(self, Gndi, sites, signals):
        self.stims = {site: signal for site, signal in zip(sites, signals)}
        # TODO build the stimulation vectors
        return Gndi

    def sim(self, inputs, nsteps):  # TODO think: now nsteps is assumed to be the same for recording, stim and sim
        outputs = []
        for s in range(nsteps):

            # TODO stimulate !!!

            outputs.append(self.cornet(inputs[s % len(inputs)]))
        for r in self.recordings.keys():
            site = self.site_mapping[r]
            for s in range(nsteps):
                self.recordings[r][s] = self.states[site[0]][s][site[1][0], site[1][1]]

        self.states = {'V1': [], 'V2': [], 'V4': [], 'IT': []}
        return self.recordings, outputs


if __name__ == '__main__':
    input_dim = (64, 64)
    neur = NeuronalModel(input_dim)
    Gndi = {s: i for i, s in enumerate(np.cumsum([d[0] * d[1] - 1 for _, d in neur.conv_dims.items()]))}
    neur.build(Gndi)  # ({'V1': 0, 'V2': 1, 'V4': 2, 'IT': 3})
    neur.cornet(torch.zeros((1, 3) + input_dim))
    print(neur.states['V1'][0].shape)
    print(neur.states['V2'][0].shape)
    print(neur.states['V4'][0].shape)
    print(neur.states['IT'][0].shape)

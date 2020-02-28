import os
import torch
from torch.utils import model_zoo
import cornet
import numpy as np
import matplotlib.pyplot as plt
from torch.nn import Module, Sequential


class NeuronalModel:
    def __init__(self):
        self.Gn = None
        self.cornet = cornet.cornet_s(pretrained=True)
        self.states = {'V1': [], 'V2': [], 'V4': [], 'IT': []}

    class OutputReader:
        def __init__(self, state_arr):
            self.state_arr = state_arr

        def __call__(self, module, input, output):
            self.state_arr.append(output)

    def build(self, Gndi):
        # TODO condition the hooks on Gndi
        self.cornet.module.V1.register_forward_hook(NeuronalModel.OutputReader(self.states['V1']))
        self.cornet.module.V2.register_forward_hook(NeuronalModel.OutputReader(self.states['V2']))
        self.cornet.module.V4.register_forward_hook(NeuronalModel.OutputReader(self.states['V4']))
        self.cornet.module.IT.register_forward_hook(NeuronalModel.OutputReader(self.states['IT']))

    def record(self, Gndi, sites, nsteps):
        pass

    def stim(self, Gndi, sites, signal):
        pass

    def sim(self, sites, signal, nsteps):
        pass

    def reset(self, ):
        pass


if __name__ == '__main__':
    neur = NeuronalModel()
    neur.build(None)
    neur.cornet(torch.zeros((1, 3, 64, 64)))
    print(neur.states['V1'])
    print(neur.states['V2'])
    print(neur.states['V4'])
    print(neur.states['IT'])

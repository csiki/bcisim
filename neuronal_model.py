import os
import torch
from torch.utils import model_zoo


class NeuronalModel:
    def __init__(self):
        self.Gn = None
        # TODO load cornet model - need to be done in linux: pip install git+https://github.com/dicarlolab/CORnet

    def build(self, Gndi):
        pass

    def record(self, Gndi, sites, nsteps):
        pass

    def stim(self, Gndi, sites, signal):
        pass

    def sim(self, sites, signal, nsteps):
        pass

    def reset(self, ):
        pass


if __name__ == '__main__':
    model = NeuronalModel()

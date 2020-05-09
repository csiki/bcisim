import numpy as np
from neuronal_model import NeuronalModel
from typing import Union, Tuple
import torch


# recording: device gets binary array of length = #recording sites
#   binary array is converted to indices, which are then mapped to neuronal model indices
#   neur maps those indices to the actual neurons;
#   the simulation returns the recorded activities in the order of the neuronal model indices provided to neur
# stimulation: device gets a stimulation array of nsteps x batch x ndevsites
#   same as recording, but device first produces indices of the stimulated sites and separate stim signals
class Device:
    def __init__(self, neur: NeuronalModel, hwdev):
        self.neur = neur
        self.hwdev = hwdev  # torch device
        self.Gd = None  # should be initialized here if the device have a particular geometry
        self.Gndi = None
        self.dev2neur, self.dev2neur_vec = None, None
        self.active_devsites, self.available_neursites = None, None
        self.Ar = None

    def implant(self, Gnd: dict):
        # Gnd is neur->dev connection graph or dictionary (dict implemented for now)
        self.Gndi = self.neur.build(Gnd)

        # build a reverse Gndi; active_devsites[i] corresponds to available_neursites[i]
        self.dev2neur = {d: n for n, d in self.Gndi.items()}
        self.dev2neur_vec = np.vectorize(self.dev2neur.__getitem__)  # gives speedup when mapping dev to neur sites
        self.active_devsites = np.array(sorted(self.dev2neur.keys()))
        self.available_neursites = self.dev2neur_vec(self.active_devsites)
        self.active_devsites = np.array(self.active_devsites)
        self.available_neursites = np.array(self.available_neursites)

    def record(self, Ar: Union[torch.Tensor]):
        # Ar is a binary vector indexed by (active) device recording sites, multiple sites can be 1 at the same time
        #   Ar: nsteps x batch x ndevsites
        # sum(Ar, axis=-1) should be <= nreclim (number of sites the device can record simultaneously)
        # generate the neur recording sites vector of indices according to dev2neur
        # "record" from all sites first, then filter/mask it later in read()
        self.Ar = Ar  # save Ar to mask recorded activity in read()

        # all available sites for all steps and batches
        sites = np.tile(self.available_neursites, Ar.shape[:-1] + (1,))
        # sites = self.available_neursites.repeat(Ar.shape[:-1] + (1,))
        self.neur.record(sites)

    def stim(self, As: Union[torch.Tensor]):
        # As is a signal vector indexed by device recording sites, contains all the stimulation signals
        # As: nsteps x batch x ndevsites
        # 0 means no stimulation
        sites = np.tile(self.available_neursites, As.shape[:-1] + (1,))
        # sites = self.available_neursites.repeat(As.shape[:-1] + (1,))  # all available sites
        signals = As[:, :, self.active_devsites]  # remove dev sites that don't have corresponding neur sites
        self.neur.stim(sites, signals)

    def read(self, activity: Tuple[torch.Tensor, torch.Tensor]):
        # fill in recordings for dev sites that are not connected to any neuron (could be all 0s or random)
        # use Ar to mask out activity not deemed to be recorded
        recordings, output = activity
        tru_recordings = torch.rand_like(self.Ar, device=self.hwdev)  # * torch.std(recordings) + torch.mean(recordings)
        tru_recordings[:, :, self.active_devsites] = recordings
        tru_recordings = tru_recordings * self.Ar  # mask out sites that were not selected for recording
        return tru_recordings, output

    def reset(self):
        pass


if __name__ == '__main__':
    hwdev = 'cuda:0'
    dev = Device(NeuronalModel((3, 64, 64)), hwdev)
    dev.implant({0: 1, 3: 4, 4: 2})

    Ar = torch.tensor([[[0, 1, 0, 1, 1]]], dtype=torch.float32, device=hwdev)
    As = torch.tensor([[[0, 1, 0, 1, 1]]], dtype=torch.float32, device=hwdev)
    dev.record(Ar)
    dev.stim(As)

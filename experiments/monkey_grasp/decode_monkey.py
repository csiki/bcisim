import numpy as np
from scipy.signal import upfirdn
import matplotlib.pyplot as plt
import os, sys
import re
import copy
import pickle
import torch
from torch.nn import Module
import torch.nn.functional as F
from torch_geometric.data import Data as Graph
from torch_geometric.utils.convert import to_networkx
import networkx as nx

from neur_dec import NeurDec


def blackrock_arraygrid(blackrock_elid_list: list, chans: set) -> np.ndarray:
    array_grid = np.zeros((10, 10), dtype=int) - 1
    for i in range(10):
        for j in range(10):
            idx = (9 - i) * 10 + j
            bl_id = blackrock_elid_list[idx]
            array_grid[i, j] = bl_id

    arr = np.ma.array(array_grid, mask=np.isnan(array_grid))

    # remove channels from grid that is not present in the set of channels
    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            arr[i, j] = arr[i, j] if arr[i, j] in chans else -1

    return arr


def plot_graph(g: Graph):
    print(g)
    plt.figure(figsize=(14, 12))
    nx.draw(to_networkx(g), {i: p.numpy() for i, p in enumerate(g.pos)}, cmap=plt.get_cmap('Set1'),
            node_size=3, linewidths=1)
    plt.show()


class MonkeyDataGen:

    def __init__(self, monkey: dict, chan_conn: np.ndarray, fs:int):
        self.ntrials = len(monkey[b'spike_trains'])
        self.nunits = len(monkey[b'spike_trains'][0])
        self.spikes = [[chan.astype(int) for chan in trial] for trial in monkey[b'spike_trains']]  # 30 kHz; to int

        self.analog = monkey[b'analog_signals']  # 1 kHz
        an_chan_ids = [an_meta[b'channel_id'] for an_meta in monkey[b'analog_meta'][8]]
        self.an_load_force_i = an_chan_ids.index(141) if 141 in an_chan_ids else None  # pulling force
        self.an_displ_i = an_chan_ids.index(143) if 143 in an_chan_ids else None  # object displacement

        # assume that the spike meta data, and thus the ordering of electrodes in spike trains is the same across trials
        self.spike_meta = monkey[b'spike_meta'][0]
        self.chan_ids = np.array([sm[b'channel_id'] for sm in self.spike_meta], dtype=int)  # have duplicates (per unit)
        self.unit_ids = np.array([sm[b'unit_id'] for sm in self.spike_meta], dtype=int)
        self.chan_map = {cid: np.where(self.chan_ids == cid)[0] for cid in np.unique(self.chan_ids)}
        self.unit_map = {sm[b'unit_id']: i for i, sm in enumerate(self.spike_meta)}  # maps unit ids to indices

        self.grid = blackrock_arraygrid(monkey[b'blackrock_elid_list'], set(self.chan_ids))

        self.graph_zero = self.build_template_graph(chan_conn, node_dim=1)
        # plot_graph(self.graph_zero)

        # resample spikes and analog signals
        spike_fs, analog_fs = 30000, 1000
        if fs != spike_fs:  # need to resample spikes
            # multiply spike timestamps by the ratio of old and new sampling freqs, then remove timestamp duplicates
            fs_ratio = fs / spike_fs
            for trial in monkey[b'spike_trains']:
                for chan_i, chan in enumerate(trial):
                    trial[chan_i] = np.unique(chan * fs_ratio)  # timestamps should be ordered anyways

        if fs > analog_fs:  # upsample analog
            fs_ratio = fs // analog_fs
            for trial in self.analog:
                for chan_i, chan in enumerate(trial):
                    trial[chan_i] = upfirdn([.5, 1, .5], chan, up=fs_ratio, mode='edge')  # linear upsampling
        elif fs < analog_fs:
            raise NotImplemented(f'why would even want to go below {analog_fs} Hz?!')

        # TODO support events

    def build_template_graph(self, chan_conn: np.ndarray, node_dim):
        # template graph: derive connectivity of channels,
        #   then have complete graphs of units connected along channel edges
        # channel connectivity is a binary KxK matrix, representing the connectivity relatively centered at an electrode
        # K has to be odd, chan_conn[K//2, K//2] is a dontcare
        assert chan_conn.shape[0] == chan_conn.shape[1] and chan_conn.shape[0] % 2 == 1
        kernel_rad = chan_conn.shape[0] // 2

        # loop over the chan grid array and construct the edges according to chan_conn and the grid
        edges = []  # contains pairs of unit indices
        oob = lambda i, j: not (0 <= i < self.grid.shape[0] and 0 <= j < self.grid.shape[1])
        for i in range(self.grid.shape[0]):
            for j in range(self.grid.shape[1]):
                cid = self.grid[i, j]
                if cid == -1:
                    continue

                uids = self.chan_map[cid]  # all indices of units for the given channel

                # get channels (and thus units) to connect to according to chan_conn
                #   sort of convolve chan_conn on grid
                conn_uids = []
                for ki in range(chan_conn.shape[0]):
                    for kj in range(chan_conn.shape[1]):
                        gi = i - kernel_rad + ki
                        gj = j - kernel_rad + kj
                        if ki == kj or oob(gi, gj) or self.grid[gi, gj] == -1:
                            continue  # itself, out of bounds, or invalid channel
                        if chan_conn[ki, kj]:
                            conn_uids.append(self.chan_map[self.grid[gi, gj]])

                units2conn = np.concatenate(conn_uids)  # neighbors
                assert len(units2conn) == len(np.unique(units2conn))

                # connect all units with all others in the neighborhood; assume undirected graph w/o self-loops
                self_edges = np.array([[uids[u1i], uids[u2i]] for u1i in range(len(uids))
                                                              for u2i in range(u1i + 1, len(uids)) if u1i != u2i])
                neighbor_edges = np.array([[u1, u2] for u1 in uids
                                                    for u2 in units2conn if u1 != u2])
                if len(self_edges) > 0:
                    edges.append(self_edges)
                if len(neighbor_edges) > 0:
                    edges.append(neighbor_edges)

        edges = torch.from_numpy(np.concatenate(edges).T)  # shape: 2xN

        # finally get positions of each unit from the grid (=graph.pos) for debugging/plotting purposes only
        pos = torch.zeros((self.nunits, 2), dtype=torch.float32)  # 2D grid
        for i in range(self.grid.shape[0]):
            for j in range(self.grid.shape[1]):
                if self.grid[i, j] == -1:
                    continue
                units = self.chan_map[self.grid[i, j]]
                for u in units:
                    pos[u] = torch.tensor([i, j])
        pos += torch.rand_like(pos) * .3  # add a little noise to spread the units out

        return Graph(x=torch.zeros((self.nunits, node_dim)), edge_index=edges, pos=pos)

    def vec_trial_gen(self, trial_i: int, field_names: list):
        # yields spiking data as binary vectors
        unit_t = np.zeros(self.nunits)  # time anchor for each unit so unit_t[u] <= step at all times
        spikes_trial = self.spikes[trial_i]
        analog_trial = self.analog[trial_i]

        # append -1 to the end of spikes so we know where the end is
        if len(spikes_trial[0]) == 0 or spikes_trial[0][0] != -1:
            for chan_i, chan in enumerate(spikes_trial):
                spikes_trial[chan_i] = np.concatenate([chan, [-1]])

        # decode field list, catch fields like spike_t+X, force_t+X
        spikes_re = re.compile(r'spikes_t\+{0,9}')
        forces_re = re.compile(r'forces_t\+{0,9}')
        field_spikes_i = field_names.index('spikes_t')  # mandatory field
        field_forces_i = field_names.index('forces_t')  # mandatory field
        fields_future_spikes = [(int(f[f.index('_t') + 3:]), f_i)
                                for f_i, f in enumerate(field_names) if spikes_re.match(f)]
        fields_future_spikes = sorted(fields_future_spikes)  # faster to retrieve future spikes if deltas are in order
        fields_future_forces = [(int(f[f.index('_t') + 3:]), f_i)
                                for f_i, f in enumerate(field_names) if forces_re.match(f)]

        # stop when out of analog signals; if future fields, then stop sooner
        nstep = len(analog_trial[0]) - max([fut for fut, f_i in fields_future_spikes + fields_future_forces])
        for step in range(nstep):
            res = [None] * (2 + len(fields_future_spikes) + len(fields_future_forces))  # assembled fields to return

            # retrieve spikes of this timestep
            spikes_vec, unit_t = self._get_next_spikes_vec(spikes_trial, unit_t, step)

            # assign current spikes and force
            res[field_spikes_i] = spikes_vec
            res[field_forces_i] = analog_trial[self.an_load_force_i][step]

            # retrieve future spikes
            fut_step = step + 1  # already stepped one forward with unit_t
            fut_unit_t = unit_t.copy()  # copy so no prob w/ temporarily overwriting it
            for delta, f_i in fields_future_spikes:
                target_step = step + delta
                for _ in range(target_step - fut_step):  # take some steps up to target
                    _, fut_unit_t = self._get_next_spikes_vec(spikes_trial, fut_unit_t, fut_step)
                    fut_step += 1

                fut_spikes_vec, fut_unit_t = self._get_next_spikes_vec(spikes_trial, fut_unit_t, fut_step)
                res[f_i] = fut_spikes_vec  # and now the loop can continue on w/ fut_step and fut_unit_t being updated

            # retrieve future analog signals
            for delta, f_i in fields_future_forces:
                res[f_i] = analog_trial[self.an_load_force_i][step + delta]

            yield res  # TODO test this whole generator

    def _get_next_spikes_vec(self, spikes_trial: list, unit_t: np.ndarray, step: int):
        # retrieve spike times indexed by unit_t and set spike to 1 if the time is the current time (step)
        # overwrites unit_t
        spikes_t_vec = np.array([chan[unit_t[chan_i]] for chan_i, chan in enumerate(spikes_trial)])
        valid_spikes = spikes_t_vec == step

        spikes_vec = np.zeros(self.nunits)
        spikes_vec[valid_spikes] = 1
        unit_t[valid_spikes] += 1  # increment timestamp anchor

        return spikes_vec, unit_t


    def graph_trial_gen(self, trial_i: int, field_names: list):
        # TODO call vec_trial_gen and convert vectors to graphs and return that, easy
        pass

    def build_graph(self, spike_data: list):
        pass


# convert monkey neural recordings into graphs
# create a generator that outputs those graphs w/ future states and outcomes to be predicted
monkey_name = 'Lilou'  # Lilou or Nikos2
with open(f'{monkey_name}.pckl', 'rb') as f:
    monkey = pickle.load(f, encoding='bytes')

nine_neighbor = np.ones((3, 3), dtype=int)
four_neighbor = np.array([
    [0, 1, 0],
    [1, 1, 1],
    [0, 1, 0],
], dtype=int)
gen = MonkeyDataGen(monkey, four_neighbor)
print(len(gen.spikes), len(gen.spikes[0]))

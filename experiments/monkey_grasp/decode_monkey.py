import numpy as np
import matplotlib.pyplot as plt
import os, sys
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

    def __init__(self, monkey: dict, chan_conn: np.ndarray):
        self.ntrials = len(monkey[b'spike_trains'])
        self.nunits = len(monkey[b'spike_trains'][0])
        self.spikes = monkey[b'spike_trains']

        # assume that the spike meta data, and thus the ordering of electrodes in spike trains is the same across trials
        self.spike_meta = monkey[b'spike_meta'][0]
        self.chan_ids = np.array([sm[b'channel_id'] for sm in self.spike_meta], dtype=int)  # have duplicates (per unit)
        self.unit_ids = np.array([sm[b'unit_id'] for sm in self.spike_meta], dtype=int)
        self.chan_map = {cid: np.where(self.chan_ids == cid)[0] for cid in np.unique(self.chan_ids)}
        self.unit_map = {sm[b'unit_id']: i for i, sm in enumerate(self.spike_meta)}  # maps unit ids to indices

        self.grid = blackrock_arraygrid(monkey[b'blackrock_elid_list'], set(self.chan_ids))

        self.graph_zero = self.build_template_graph(chan_conn, node_dim=1)
        # plot_graph(self.graph_zero)

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

    def vec_trial_gen(self, trial_i: int):
        step = 0
        unit_t = np.zeros(self.nunits)  # time anchor for each unit so unit_t[u] <= step
        pass  # TODO same as graph but outputs binary vectors

    def graph_trial_gen(self, trial_i: int):
        # TODO call vec_trial_gen and convert vectors to graphs and return that, easy
        pass

    def build_graph(self, spikes):
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

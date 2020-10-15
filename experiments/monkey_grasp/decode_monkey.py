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

    def __init__(self, monkey: dict, chan_conn: np.ndarray, fs: int):
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

        # TODO implement collapse units here: if required, units of the same channel can be collapsed to the channel

        self.grid = blackrock_arraygrid(monkey[b'blackrock_elid_list'], set(self.chan_ids))

        self.graph_zero = self.build_template_graph(chan_conn, node_dim=1)
        # plot_graph(self.graph_zero)

        # count overall number of spikes to see how much we lose after down/upsampling
        nspikes_total = sum([len(chan) for trial in self.spikes for chan in trial])

        # resample spikes and analog signals
        spike_fs, analog_fs = 30000, 1000
        if fs != spike_fs:  # need to resample spikes
            # multiply spike timestamps by the ratio of old and new sampling freqs, then remove timestamp duplicates
            fs_ratio = fs / spike_fs
            for trial in self.spikes:
                for chan_i, chan in enumerate(trial):
                    trial[chan_i] = np.unique((chan * fs_ratio).astype(int))  # timestamps should be ordered anyways

        nspikes_total_after_resampling = sum([len(chan) for trial in self.spikes for chan in trial])
        print(f'{nspikes_total_after_resampling / nspikes_total * 100:.2f}% spikes remained after resampling')

        # upsample analog signals
        if fs > analog_fs:
            fs_ratio = fs // analog_fs
            up_kernel = np.ones(fs_ratio)  # linear upsampling
            for trial in self.analog:
                for chan_i, chan in enumerate(trial):
                    trial[chan_i] = upfirdn(up_kernel, chan.ravel(), up=fs_ratio, mode='edge').reshape((-1, 1))

        elif fs < analog_fs:
            pass
            # raise NotImplemented(f'why would even want to go below {analog_fs} Hz?!')

        # TODO support events

        # prepare regexp to capture field names like spikes_t+X or forces_t+X (e.g. in function vec_trial_gen)
        self.fut_spikes_re = re.compile(r'spikes_t\+[0-9]+')
        self.fut_forces_re = re.compile(r'forces_t\+[0-9]+')

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

    def trial_len(self, trial_i):  # only works if no t+X fields are required, otherwise trial_len() - X
        return len(self.analog[trial_i][0])  # analog defines the length, spike timestamps are unreliable

    def vec_trial_gen(self, trial_i: int, field_names: list):
        # yields spiking data as binary vectors
        unit_t = np.zeros(self.nunits, dtype=int)  # time anchor for each unit so unit_t[u] <= step at all times
        spikes_trial = self.spikes[trial_i]
        analog_trial = self.analog[trial_i]

        # append -1 to the end of spikes so we know where the end is, and the below iteration can be done w/o branching
        if len(spikes_trial[0]) == 0 or spikes_trial[0][-1] != -1:
            for chan_i, chan in enumerate(spikes_trial):
                spikes_trial[chan_i] = np.concatenate([chan, [-1]])

        # decode field list, catch fields like spike_t+X, force_t+X
        field_spikes_i = field_names.index('spikes_t')  # mandatory field
        field_forces_i = field_names.index('forces_t')  # mandatory field
        fields_future_spikes = [(int(f[f.index('_t') + 3:]), f_i)
                                for f_i, f in enumerate(field_names) if self.fut_spikes_re.match(f)]
        fields_future_spikes = sorted(fields_future_spikes)  # faster to retrieve future spikes if deltas are in order
        fields_future_forces = [(int(f[f.index('_t') + 3:]), f_i)
                                for f_i, f in enumerate(field_names) if self.fut_forces_re.match(f)]

        # stop when out of analog signals; if future fields, then stop sooner
        nstep = self.trial_len(trial_i) - max([0] + [fut for fut, _ in fields_future_spikes + fields_future_forces])
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
        # TODO call in graph_trial_gen
        pass


if __name__ == '__main__':
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

    fs = 1000
    gen = MonkeyDataGen(monkey, four_neighbor, fs)

    # build training and test sets
    train_ratio = .7
    fields = ['spikes_t', 'forces_t']

    trial_indices = np.random.permutation(gen.ntrials)
    train_trials = set(trial_indices[:int(gen.ntrials * train_ratio)])
    test_trials = set(trial_indices[len(train_trials):])

    train_spikes, train_forces = [], []
    test_spikes, test_forces = [], []
    for trial_i in trial_indices:
        for i, s in enumerate(gen.vec_trial_gen(trial_i, fields)):
            if trial_i in train_trials:
                train_spikes.append(s[0])
                train_forces.append(s[1])
            else:
                test_spikes.append(s[0])
                test_forces.append(s[1])
    train_spikes, train_forces = np.array(train_spikes), np.array(train_forces)
    test_spikes, test_forces = np.array(test_spikes), np.array(test_forces)
    print('train', train_spikes.shape, train_forces.shape)
    print('test', test_spikes.shape, test_forces.shape)

    # # train linear regression - shit
    # from sklearn.linear_model import LinearRegression
    # model = LinearRegression(n_jobs=4).fit(train_spikes, train_forces.ravel())
    # train_score = model.score(train_spikes, train_forces.ravel())
    # test_score = model.score(test_spikes, test_forces.ravel())
    # print('linear train score', train_score, 'test score', test_score)

    # # lasso - shit
    # from sklearn.linear_model import LassoLars
    # model = LassoLars().fit(train_spikes, train_forces.ravel())
    # train_score = model.score(train_spikes, train_forces.ravel())
    # test_score = model.score(test_spikes, test_forces.ravel())
    # print('lasso train score', train_score, 'test score', test_score)

    # # simple NN - shit and slow
    # from sklearn.neural_network import MLPRegressor
    # model = MLPRegressor(hidden_layer_sizes=[64], activation='tanh', early_stopping=True, validation_fraction=.2) \
    #     .fit(train_spikes, train_forces.ravel())
    # train_score = model.score(train_spikes, train_forces.ravel())
    # test_score = model.score(test_spikes, test_forces.ravel())
    # print('simple NN train score', train_score, 'test score', test_score)

    # TODO kalman filter
    from pykalman import KalmanFilter, UnscentedKalmanFilter
    model = KalmanFilter(n_dim_state=train_forces.shape[-1], n_dim_obs=train_spikes.shape[-1])
    model.em(train_spikes, train_forces, n_iter=10)
    smoothed_state_means, smoothed_state_covariances = model.smooth(test_spikes)

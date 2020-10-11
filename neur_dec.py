import os, sys
from typing import List
import copy
from queue import Queue

import torch
from torch.nn import Module
import torch.nn.functional as F
from torch_geometric.data import Data as Graph
from torch_geometric.nn import GATConv, NNConv
# import numpy as np
# import matplotlib.pyplot as plt


class InferBranch(Module):
    pass


class SelfSupBranch(Module):  # self-supervised binary prediction for each node
    def __init__(self, in_chan: int, pred_chan: int = 1):
        super().__init__()
        self.lin = torch.nn.Linear(in_chan, pred_chan)

    def forward(self, g):
        x = self.lin(g.x)
        return F.sigmoid(x)


class NeurDec(Module):

    def __init__(self, graph_init: Graph, in_chan: int, h_chan: int, h_edge_dim: int, branches: List[Module], **kwargs):
        super().__init__()
        self.device = torch.device(getattr(kwargs, 'device', 'cuda' if torch.cuda.is_available() else 'cpu'))

        self.h = copy.deepcopy(graph_init).to(self.device)
        assert self.h.x.shape[1] == h_chan

        self.in_chan = in_chan
        self.h_chan = h_chan
        self.h_edge_dim = h_edge_dim
        self.branches = branches

        # TODO could try: NNConv, GATConv, LayerNorm
        att_head = getattr(kwargs, 'att_heads', 1)
        dropout = getattr(kwargs, 'dropout', 0)
        self.gnn_x = GATConv(in_chan, h_chan, att_head, dropout=dropout)
        self.gnn_h = GATConv(h_chan * 2, h_chan, att_head, dropout=dropout)

    @torch.no_grad()
    def reset(self):
        self.h.x = torch.zeros_like(self.h.x).to(self.device)

    def detach(self):
        self.h.x.detach_()

    def forward(self, g):
        x, edge_index = g.x, g.edge_index
        x = self.gnn_x(x, edge_index)

        x = torch.cat([self.h.x, x], dim=-1)
        self.h.x = self.gnn_h(x, self.h.edge_index)

        return [branch(self.h) for branch in self.branches]


def test_net():

    torch.autograd.set_detect_anomaly(True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # gen seq of graphs
    seq_len = 256
    edge_index = torch.tensor([[0, 1, 1, 2],
                               [1, 0, 2, 1]], dtype=torch.long).to(device)
    x = torch.tensor([[1], [0], [1]], dtype=torch.float)
    g0 = Graph(x=x, edge_index=edge_index)
    g = g0
    g_seq = []
    shift = lambda x: torch.roll(x, shifts=1, dims=0)
    for s in range(seq_len):
        g = Graph(x=shift(g.x), edge_index=edge_index)
        if len(g_seq) > 0:
            g_seq[-1].y = g.x  # y_t = x_{t-1}
        g_seq.append(g)
    g_seq[-1].y = shift(g.x)
    g_seq = [g.to(device) for g in g_seq]
    print('data generated')

    # init network
    h_chan = 4
    g_init = copy.deepcopy(g0)
    g_init.x = torch.zeros((g_init.x.shape[0], h_chan))

    branches = [SelfSupBranch(h_chan).to(device)]
    model = NeurDec(g_init, 1, h_chan, 0, branches, device=device).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-3)

    # train net
    k1 = k2 = 4
    nepoch = 10
    model.train()
    for epoch in range(nepoch):

        preds, targets = [], []
        for s in range(seq_len):
            pred = model(g_seq[s])[0]
            target = g_seq[s].y

            preds.append(pred)
            targets.append(target)

            # TBPTT
            if (s + 1) % k1 == 0:
                optimizer.zero_grad()
                loss = F.binary_cross_entropy(torch.stack(preds), torch.stack(targets))
                loss.backward(retain_graph=False)
                optimizer.step()
                model.detach()  # detach hidden state to cut backprop-graph

                preds.clear()
                targets.clear()

        model.reset()
        print(f'epoch {epoch + 1}/{nepoch}')

    # test
    model.eval()
    preds = torch.stack([model(g_seq[s])[0] for s in range(seq_len)])
    ytrue = torch.stack([g_seq[s].y for s in range(seq_len)])

    print('error:', torch.mean(preds - ytrue))


if __name__ == '__main__':
    test_net()

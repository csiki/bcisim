import numpy as np
import sys, os
import torch
from torch.nn import Module, LSTM, Sequential, Linear, MSELoss, Conv2d
from collections import OrderedDict


class SimpleLSTM(Module):  # embedding may be some conv layers if input is conv
    def __init__(self, input_dim: int, hidden_dim: int, out_dim: int, num_layers: int, embedding: Module = None):
        super().__init__()
        self.embedding = embedding
        self.lstm = LSTM(input_size=input_dim, hidden_size=hidden_dim, num_layers=num_layers,
                         batch_first=True, dropout=.3)
        self.lin = Linear(hidden_dim, out_dim)

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.h, self.c = None, None

    def init_hidden(self, batch_size: int):
        dev = next(self.parameters()).device
        return (torch.rand((self.num_layers, batch_size, self.hidden_dim), device=dev),
                torch.randn((self.num_layers, batch_size, self.hidden_dim), device=dev))

    def forward(self, x):
        x = self.embedding(x) if self.embedding else x
        x, (self.h, self.c) = self.lstm(x, (self.h, self.c))
        return self.lin(x)

    def tbptt_train(self, trial_gens, loss_fun, optimizer, k: int):
        # TODO implement adaptive lr

        self.train()
        dev = next(self.parameters()).device
        ntrials = len(trial_gens)  # same as batch size now
        self.h, self.c = self.init_hidden(batch_size=ntrials)

        out_of_seq = False
        losses = []
        while not out_of_seq:
            batch_x, batch_y = [], []
            try:
                for trial_gen in trial_gens:
                    seq = [next(trial_gen) for _ in range(k)]
                    batch_x.append([x for x, y in seq])
                    batch_y.append([y for x, y in seq])

                batch_x = torch.tensor(batch_x, dtype=torch.float32, device=dev)  # ntrial x k x feat,
                batch_y = torch.tensor(batch_y, dtype=torch.float32, device=dev)  # where ntrial == batch size

                # actual training
                self.zero_grad()
                self.h.detach_()
                self.c.detach_()
                self.h, self.c = self.h.detach(), self.c.detach()
                y_pred = self.forward(batch_x)
                loss = loss_fun(y_pred, batch_y)
                loss.backward()
                optimizer.step()
                losses.append(loss.detach().cpu())

            except (RuntimeError, StopIteration):
                print('k is off', file=sys.stderr)  # TODO handle when trial_len % k != 0 for all trials
                break

        return np.mean(losses)

    def test(self, trial_gens, loss_fun):
        dev = next(self.parameters()).device
        with torch.no_grad():
            self.eval()
            self.h, self.c = self.init_hidden(batch_size=1)  # testing each trial separately, no batching
            losses = []
            for trial_gen in trial_gens:
                samples = [sample for sample in trial_gen]
                x = torch.tensor([sample[0] for sample in samples], dtype=torch.float32, device=dev)
                y = torch.tensor([sample[1] for sample in samples], dtype=torch.float32, device=dev)
                x, y = torch.unsqueeze(x, 0), torch.unsqueeze(y, 0)  # add batch dim
                y_pred = self(x)
                losses.append(loss_fun(y_pred, y).cpu())

            return losses


class Reshape(Module):
    def __init__(self, view):
        super().__init__()
        self.view = view

    def forward(self, x: torch.Tensor):
        return x.view(self.view)


class GridConv(Module):
    def __init__(self, chans=[4, 8, 8, 8]):
        super().__init__()
        self.chans = chans

        self.grid_embedding = Sequential(OrderedDict([
            # ('gridify', Reshape((-1, 1, 10, 10))),
            ('conv1', Conv2d(in_channels=1, out_channels=chans[0], kernel_size=(3, 3), stride=1, padding=1)),  # 4x10x10
            ('conv2', Conv2d(in_channels=chans[0], out_channels=chans[1], kernel_size=(3, 3), stride=2, padding=0)),  # 8x4x4
            ('conv3', Conv2d(in_channels=chans[1], out_channels=chans[2], kernel_size=(3, 3), stride=2, padding=1)),  # 8x2x2
            ('conv4', Conv2d(in_channels=chans[2], out_channels=chans[3], kernel_size=(3, 3), stride=2, padding=1)),  # 8x1x1
            # ('flatten', Reshape((-1, k, 8))),
        ]))

    def forward(self, x):
        time_dim = x.shape[1]
        x = self.grid_embedding(x.view((-1, 1, 10, 10)))
        return x.view((-1, time_dim, self.chans[-1]))
